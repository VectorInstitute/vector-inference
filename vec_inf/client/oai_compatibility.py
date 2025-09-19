"""OpenAI-compatible wrapper for vLLM servers launched through vec-inf."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

import backoff
import httpx
from openai import AsyncOpenAI, OpenAIError
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from vec_inf.client.api import VecInfClient
from vec_inf.client.models import LaunchOptions, ModelStatus


MODEL_NAME = "Qwen3-0.6B"
VLLM_ARGS = "--enable-auto-tool-choice --tool-call-parser hermes"
EXAMPLE_MESSAGES: list[ChatCompletionMessageParam] = [
    {"role": "user", "content": "Introduce yourself."}
]
LAUNCH_OPTIONS = LaunchOptions(vllm_args=VLLM_ARGS, qos="scavenger")

T = TypeVar("T")


if TYPE_CHECKING:
    AsyncOpenAICompatible = AsyncOpenAI
else:
    class _AsyncCompletionsCompatible(Protocol):
        def create(self, *args: object, **kwargs: object) -> Awaitable[ChatCompletion]: ...

    class _AsyncChatCompatible(Protocol):
        completions: _AsyncCompletionsCompatible

    class AsyncOpenAICompatible(Protocol):
        """Structural subset of AsyncOpenAI for static type checkers."""

        chat: _AsyncChatCompatible


@dataclass
class _ReplicaState:
    slot: int
    slurm_job_id: int
    base_url: str | None = None
    status: ModelStatus = ModelStatus.PENDING


class AsyncVecInf:
    """Async Context Manager for managing vec-inf jobs.

    Example usage:
    ```
    async with AsyncVecInf(
        model=MODEL_NAME,
        num_replicas=2,
        options=LAUNCH_OPTIONS,
    ) as async_oai_client:
        coros = [
            async_oai_client.chat.completions.create(
                messages=EXAMPLE_MESSAGES, model=MODEL_NAME
            )
            for _ in range(18)
        ]
        responses = await asyncio.gather(coros)
        print(responses[0].model_dump_json(indent=2))
    ```

    The context manager:
    - spins up num_replicas jobs using vec_inf_client.launch_model
    - exposes an OpenAI-compatible async_oai_client duck-typing
        the chat completion interface
    - upon requests, the OpenAI-compatible client blocks (asynchronously)
        any request until at least one vec_inf job is ready.
    - if more than one jobs are "ready" (using vec_inf_client.get_status)
        route the request to the jobs randomly.
    - check status of jobs in intervals- if a job goes down, replace that job.
    - use exponential backoff from `backoff` when forwarding requests.
    - invokes shutdown_model on all instances (ready or not) on context manager exit.
    """

    def __init__(
        self,
        model: str,
        num_replicas: int,
        options: LaunchOptions | None = None,
        status_poll_interval: float = 30.0,
    ) -> None:
        if num_replicas < 1:
            raise ValueError("num_replicas must be at least 1")

        self._model: str = model
        self._num_replicas: int = num_replicas
        self._options: LaunchOptions | None = options
        self._status_poll_interval: float = status_poll_interval
        self._client: VecInfClient = VecInfClient()
        self._condition: asyncio.Condition = asyncio.Condition()
        self._shutdown: bool = False

        self._slot_states: dict[int, _ReplicaState] = {}
        self._ready_slots: dict[int, str] = {}
        self._job_to_slot: dict[int, int] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._launched_job_ids: set[int] = set()

        self.logger: logging.Logger = logging.getLogger(__name__)
        self.chat: _ChatProxy = _ChatProxy(self)

    async def __aenter__(self) -> AsyncOpenAICompatible:
        """Start background tasks that maintain the requested replicas."""
        self._shutdown = False
        for slot in range(self._num_replicas):
            task = asyncio.create_task(
                self._manage_slot(slot), name=f"vec-inf-slot-{slot}"
            )
            self._tasks.append(task)
        return cast(AsyncOpenAICompatible, cast(object, self))

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop management tasks and tear down all launched jobs."""
        self._shutdown = True

        for task in self._tasks:
            _ = task.cancel()

        _ = await asyncio.gather(*self._tasks, return_exceptions=True)

        # Best-effort shutdown of every job we launched.
        shutdown_coros: list[Awaitable[None]] = []
        for job_id in sorted(self._launched_job_ids):
            shutdown_coros.append(asyncio.to_thread(self._safe_shutdown_model, job_id))

        if shutdown_coros:
            _ = await asyncio.gather(*shutdown_coros, return_exceptions=True)

        self._tasks.clear()

    @property
    def launched_job_ids(self) -> tuple[int, ...]:
        """Return the sorted set of SLURM job IDs managed by the context."""
        return tuple(sorted(self._launched_job_ids))

    async def _manage_slot(self, slot: int) -> None:
        self.logger.info("Creating slot %s", slot)
        try:
            while not self._shutdown:
                job_id: int | None = None
                try:
                    job_id = await self._launch_replica(slot)
                    base_url = await self._await_replica_ready(slot, job_id)
                    should_restart = await self._monitor_replica(slot, job_id, base_url)
                    if not should_restart:
                        return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning(
                        "Slot %s encountered an error while managing job %s: %s",
                        slot,
                        job_id,
                        exc,
                    )
                    if job_id is not None:
                        await self._teardown_replica(slot, job_id)

                if not self._shutdown:
                    await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass

    async def _launch_replica(self, slot: int) -> int:
        launch_response = await asyncio.to_thread(
            self._client.launch_model, self._model, self._options
        )
        job_id = launch_response.slurm_job_id
        self._launched_job_ids.add(job_id)
        state = _ReplicaState(slot=slot, slurm_job_id=job_id)
        self._slot_states[slot] = state
        self._job_to_slot[job_id] = slot
        self.logger.info("Slot %s launched job %s", slot, job_id)
        return job_id

    async def _await_replica_ready(self, slot: int, job_id: int) -> str:
        try:
            ready_status = await self._client.wait_until_ready(job_id)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Slot %s failed to start job %s: %s", slot, job_id, exc)
            await self._teardown_replica(slot, job_id)
            raise

        base_url = ready_status.base_url
        if not base_url:
            self.logger.warning(
                "Slot %s received ready status without base_url for job %s",
                slot,
                job_id,
            )
            await self._teardown_replica(slot, job_id)
            raise RuntimeError("Replica became ready without a base URL")

        await self._mark_ready(slot, job_id, base_url)
        self.logger.info("Slot %s job %s ready at %s", slot, job_id, base_url)
        return base_url

    async def _monitor_replica(self, slot: int, job_id: int, base_url: str) -> bool:
        try:
            while not self._shutdown:
                await asyncio.sleep(self._status_poll_interval)
                if self._shutdown:
                    return False

                state_snapshot = self._slot_states.get(slot)
                if not state_snapshot or state_snapshot.status != ModelStatus.READY:
                    await self._teardown_replica(slot, job_id)
                    return True

                status = await self._client.get_status(job_id)
                if status.server_status != ModelStatus.READY or not status.base_url:
                    self.logger.info(
                        "Slot %s detected job %s unhealthy (status=%s)",
                        slot,
                        job_id,
                        status.server_status,
                    )
                    await self._teardown_replica(slot, job_id)
                    return True

                if status.base_url != base_url:
                    base_url = status.base_url
                    await self._mark_ready(slot, job_id, base_url)
                    self.logger.info(
                        "Slot %s job %s base URL updated to %s",
                        slot,
                        job_id,
                        base_url,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "Slot %s failed while polling status for job %s: %s",
                slot,
                job_id,
                exc,
            )
            await self._teardown_replica(slot, job_id)
            return True

        return False

    async def _teardown_replica(self, slot: int, job_id: int) -> None:
        await self._mark_not_ready(slot, job_id)
        await asyncio.to_thread(self._safe_shutdown_model, job_id)

    async def _mark_ready(self, slot: int, job_id: int, base_url: str) -> None:
        async with self._condition:
            state = self._slot_states.get(slot)
            if state:
                state.base_url = base_url
                state.status = ModelStatus.READY
            else:
                self._slot_states[slot] = _ReplicaState(
                    slot=slot,
                    slurm_job_id=job_id,
                    base_url=base_url,
                    status=ModelStatus.READY,
                )

            self._ready_slots[slot] = base_url
            self._job_to_slot[job_id] = slot
            self._condition.notify_all()
        self.logger.info("Slot %s job %s marked ready at %s", slot, job_id, base_url)

    async def _mark_not_ready(self, slot: int, job_id: int) -> None:
        async with self._condition:
            state = self._slot_states.get(slot)
            if state:
                state.status = ModelStatus.FAILED
                state.base_url = None
            _ = self._ready_slots.pop(slot, None)
            _ = self._job_to_slot.pop(job_id, None)

    async def _acquire_ready_replica(self) -> tuple[int, int, str]:
        async with self._condition:
            while not self._ready_slots and not self._shutdown:
                _ = await self._condition.wait()

            if self._shutdown:
                raise RuntimeError("AsyncVecInf is shutting down")

            slot = random.choice(tuple(self._ready_slots.keys()))
            base_url = self._ready_slots[slot]
            job_id = self._slot_states[slot].slurm_job_id
            return slot, job_id, base_url

    async def handle_request(
        self,
        request_fn: Callable[[AsyncOpenAI], Awaitable[T]],
    ) -> T:
        """Dispatch a request to a ready replica via the OpenAI adapter."""
        slot, job_id, base_url = await self._acquire_ready_replica()

        try:
            async with AsyncOpenAI(base_url=base_url, api_key="EMPTY") as client:
                return await request_fn(client)
        except (httpx.HTTPError, OpenAIError):
            await self._mark_not_ready(slot, job_id)
            raise

    def _safe_shutdown_model(self, job_id: int) -> None:
        with suppress(Exception):
            self.logger.info("Shutting down job %s", job_id)
            _ = self._client.shutdown_model(job_id)


class _ChatCompletionsProxy:
    def __init__(self, parent: AsyncVecInf) -> None:
        self._parent: AsyncVecInf = parent

    @backoff.on_exception(
        backoff.expo,
        (httpx.HTTPError, OpenAIError),
        max_time=60,
        jitter=backoff.full_jitter,
    )
    async def create(self, *args: object, **kwargs: object) -> ChatCompletion:
        return await self._parent.handle_request(
            lambda client: client.chat.completions.create(*args, **kwargs)
        )


class _ChatProxy:
    def __init__(self, parent: AsyncVecInf) -> None:
        self.completions: _ChatCompletionsProxy = _ChatCompletionsProxy(parent)


async def main() -> None:
    """Launch multiple replicas and exercise the OpenAI-compatible endpoint."""
    async with AsyncVecInf(
        model=MODEL_NAME,
        num_replicas=2,
        options=LAUNCH_OPTIONS,
    ) as async_oai_client:
        coros: list[Awaitable[ChatCompletion]] = [
            async_oai_client.chat.completions.create(
                messages=EXAMPLE_MESSAGES,
                model=MODEL_NAME,
            )
            for _ in range(65536)
        ]

        responses: list[ChatCompletion] = await asyncio.gather(*coros)
        print(responses[0].model_dump_json(indent=2))

        vecinf_client = cast(AsyncVecInf, cast(object, async_oai_client))
        print(f"Launched job IDs: {vecinf_client.launched_job_ids}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
