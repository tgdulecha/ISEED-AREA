# ISEED-AREA
ISEED-AREA (ARgument Extractor and Analyzer) is a research tool that can extract and analyze arguments from online debates. It contains a set of functions for extracting causal statements and other forms of argumentation from (noisy) textual data, like short posts from social media. It includes two types of argument extractions:

## How To Run
1. Install `virtualenv`:
```
$ pip install virtualenv
```

2. Open a terminal in the project root directory and run:
```
$ virtualenv iseedenv
```

3. Then run the command:
```
$ .\iseedenv\Scripts\activate
or iseedenv/bin/activate for linux/ubuntu

4. Then install the dependencies:
```
$ (env) pip install -r requirements.txt
```

5. Finally start the web server:

uvicorn app:myfastapi --reload

This server will start on port 8000 by default.

