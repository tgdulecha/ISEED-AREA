# ISEED-AREA


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

