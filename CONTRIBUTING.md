# How to Contribute

We'd love to accept your patches and contributions to this project.  We do have
some guidelines to follow, covered in this document, but don't worry about (or
expect to) get everything right the first time!  Create a pull request and
we'll nudge you in the right direction.

## Contributor License Agreement

Contributions to this project must be accompanied by a [NASA Contributor
License Agreement](/docs/nasa-cla/).

## Pull Request Process and Code Review

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose.  [GitHub
Help](https://help.github.com/articles/about-pull-requests/) has information on
using pull requests.

The preferred manner for submitting pull requests is for users to fork the
HybridQ [repo](https://github.com/nasa/hybridq) and then use a branch from this
fork to create a pull request to the main HybridQ repo.

The basic process for setting up a fork is 1. Fork the HybridQ repo (Fork
button in upper right corner of [repo page](https://github.com/nasa/hybridq)).
Forking creates a new github repo at the location
```https://github.com/USERNAME/hybridq``` where ```USERNAME``` is your github
id. Use the directions on the [development page](docs/development.md) to
download a copy to your local machine. You need only do this once.

1. Checkout main and create a new branch:
    ```shell
    git checkout main -b new_branch_name
    ```
    where ```new_branch_name``` is the name of your new branch.
2. Do your work and commit your changes to this branch.
3. If you have drifted out of sync from the main HybridQ repo
   you may need to merge in changes.  To do this, first update your local
   main and then merge the local main into your branch:
    ```shell
    # Update your local main.
    git fetch upstream
    git checkout main
    git merge upstream/main
    # Merge local main into your branch.
    git checkout new_branch_name
    git merge main
    ```
   You may need to fix merge conflicts for both of these merge
   commands.
4. Finally, push your change to your clone
    ```shell
    git push origin new_branch_name
    ```
5. Now when you navigate to the HybridQ page on github,
   [https://github.com/nasa/hybridq](https://github.com/nasa/hybridq) you
   should see the option to create a new pull request from your clone
   repository.  Alternatively you can create the pull request by navigating to
   the "Pull requests" tab in the page, and selecting the appropriate branches.
6. The reviewer will comment on your code and may ask for changes, you can
   perform these locally, and then push the new commit following the same
   process as above.

## Code Testing Standards

When a pull request is created or updated, various automatic checks will run to
ensure that the change won't break HybridQ and meets our coding standards.

Please be aware of the following code standards that will be applied to any
new changes.

- **Tests**.
Existing tests must continue to pass (or be updated) when new changes are introduced.
We use [pytest](https://docs.pytest.org/en/latest/) to run our Python tests.
- **Formatting**.
Code should meet common style standards for python and C++, and be free of
error-prone constructs. We use [YAPF](https://github.com/google/yapf) and
[clang-format](https://clang.llvm.org/docs/ClangFormat.html) to format files.
