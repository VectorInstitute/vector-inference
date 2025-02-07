"""Calculate the average prompt and generation throughput from a log file."""

import argparse
import re


def filter_throughput(log_file_path: str) -> None:
    """Filter log file for non-zero entries and calculate the avg throughput."""
    avg_prompt_throughput = []
    avg_generation_throughput = []
    # Define a regular expression pattern to extract throughput values
    pattern = r"Avg prompt throughput: ([^,]+) tokens/s, Avg generation throughput: ([^,]+) tokens/s"

    # Open the log file
    with open(log_file_path, "r") as file:
        # Iterate over each line in the file
        for line in file:
            # Use regex to find matches
            match = re.search(pattern, line)
            if match:
                # Extract prompt and generation throughput values
                prompt_throughput = match.group(1).strip()
                generation_throughput = match.group(2).strip()

                # Check if both throughput values are not zero
                if prompt_throughput != "0.0":
                    avg_prompt_throughput.append(float(prompt_throughput))
                if generation_throughput != "0.0":
                    avg_generation_throughput.append(float(generation_throughput))

    print(
        f"Average prompt throughput: {sum(avg_prompt_throughput) / len(avg_prompt_throughput)} tokens/s"
    )
    print(
        f"Average generation throughput: {sum(avg_generation_throughput) / len(avg_generation_throughput)} tokens/s"
    )


def main() -> None:
    """Run the main function."""
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Filter log file for non-zero throughput entries."
    )

    # Add the arguments
    parser.add_argument("--path", type=str, help="The path to the log file")

    # Execute the parse_args() method
    args = parser.parse_args()

    # Use the provided arguments
    filter_throughput(args.path)


if __name__ == "__main__":
    main()
