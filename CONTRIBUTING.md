# Contributing to PlanetMapper

Thanks for thinking of contributing to PlanetMapper!

## Reporting bugs & suggesting new features

If you find a bug in PlanetMapper, [open a new issue on GitHub](https://github.com/ortk95/planetmapper/issues/new). Please include as much detail as possible, including:
- A clear description of the bug, including what you expected to happen and what actually happened
- Any simple code that can reproduce the bug
- Any error messages you received
- Your operating system, Python version and PlanetMapper version

If you have an idea for a new feature, you can also [open a new issue on GitHub](https://github.com/ortk95/planetmapper/issues/new) to discuss it. Please include a clear description of the feature, and why you think it would be useful. We can't guarantee that we will implement it, but we will consider it, especially if it's related to PlanetMapper's core functionality! If you are able to implement the feature yourself, please feel free to open a pull request instead (see below).


## Contributing code

If you would like to contribute code to PlanetMapper, you are welcome to fork the repository and open a pull request. Pull requests fixing bugs, improving performance or adding features relating to the core functionality of PlanetMapper are more likely to be accepted - you can also open an issue to discuss your idea before you start coding.

Each individual pull request should be limited to a single new feature or bug fix, and include a clear description of the change. Please make sure your code is well documented and includes tests for any new functionality.

Contributions must pass all checks before they will be merged, and conform to the the style of existing code. You can run the checks locally using `run_ci.sh` - this will ensure that:
- All code is formatted using `black` and `isort`
- The code passes `pylint` and `pyright` checks
- The code passes all tests

