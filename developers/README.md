## TrojAI Developer Guidelines

### Introduction
This README contains any information relevant to developers for the `trojai` repository.

### Testing
Unit tests for the `datagen` submodule are located in the `trojai/test/datagen` directory.  To run all the unittests, first install the dependencies specified in `test_requirements.txt`.  Then, from the base project directory, run:

```
>> nosetests
```

### Requirements
`requirements.txt` contains a list of dependencies for the `TrojAI` project.
`test_requirements.txt` contains a list of dependencies to run the unittests for the `TrojAI` project.  

### Code Style Guidelines
The coding style is specified in the `intellij-java-google-style.xml` file. This can be imported by PyCharm, and perhaps other Python IDE's to provide real-time linting.