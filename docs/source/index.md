---
hide-toc: true
---

# Vector AI Engineering template repository

```{toctree}
:hidden:

user_guide
api

```

This template repository can be used to bootstrap AI Engineering project repositories
on Github! The template is meant for python codebases since Python is the most commonly
used language by our team.

The template includes:

- [pyproject.toml](https://pip.pypa.io/en/stable/reference/build-system/pyproject-toml/)
file to specify repository information and manage dependencies using
[Poetry](https://python-poetry.org/).

- [README.md](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes) which should have basic information on why the project is
useful, installation instructions and other information on how users can get started.

- [.pre-commit-config.yaml](https://pre-commit.com/) for running pre-commit hooks that
check for code-style, apply formatting, check for type hints and run tests.

- [.github/pull_request_template.md](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/creating-a-pull-request-template-for-your-repository) for PRs.

- [.github/ISSUE_TEMPLATE](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/configuring-issue-templates-for-your-repository) for bug reports and issues that can be raised on the repository.

- [.github/workflows](https://docs.github.com/en/actions/using-workflows) for running CI
workflows using Github actions. The template includes CI workflows for code checks,
documentation building and releasing python packages to PyPI.

- [LICENSE.md](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository) for adding a license to the project repository.
By default, this is the [Apache-2.0 license](http://www.apache.org/licenses/). Please
change according to your project!

- [docs](https://pradyunsg.me/furo/) for adding project documentation. Typically
projects should have API reference documentation, user guides and tutorials.

- [CONTRIBUTING.md](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/setting-guidelines-for-repository-contributors) with basic guidelines on how others can
contribute to the repository.

- [CODE_OF_CONDUCT.md](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/adding-a-code-of-conduct-to-your-project) with standards on how the community engages in
a healthy and constructive manner.

- [.gitignore](https://docs.github.com/en/get-started/getting-started-with-git/ignoring-files)
with some standard file extensions to be ignored by git. Please add/modify as necessary.

- [codecov.yml](https://docs.codecov.com/docs/codecov-yaml) for using codecov.io to
generate code coverage information for your repository. You would need to add codecov.io
app as an [integration to your repository](https://docs.codecov.com/docs/how-to-create-a-github-app-for-codecov-enterprise).


If you are starting a new project, you can navigate to the [Use this template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template) button
on the top right corner of the [template repository home page](https://github.com/VectorInstitute/aieng-template)
which will allow you to bootstrap your project repo using this template.

Please check out the user guide page for more detailed information on using the
template features. For exisiting projects, the [user guide](user_guide.md)
can be followed to migrate to following the template more closely.
