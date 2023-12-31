from flask import Flask, render_template, url_for, request, redirect, flash, session, send_file, json

import os
import json
import plotly
import plotly.express as px
import random
import string
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import re
from werkzeug.utils import secure_filename
from gensim.parsing.preprocessing import remove_stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from joblib import parallel, delayed
from flask_session import Session
from fastapi import FastAPI, Request, Path, Query, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import aiofiles
import csv
from fastapi.responses import StreamingResponse # Add to Top
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.3.1" # change as needed

UPLOAD_FOLDER = os.path.join('static', 'uploads')
TEMPAPIDATA_UPLOADFOLDER=os.path.join('static', 'APIFILES')
templates = Jinja2Templates(directory="templates/docs")
file=''

# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}
myfastapi=FastAPI(title="ISEED-API", swagger_ui_parameters={"defaultModelsExpandDepth": -1},
description=" This is an OpenAPI developed for the [ISEED project](https://iseedeurope.eu/).  You can use it to test the functions developed for the ISEED's Argument extractor. It includes two parts: **relation extractor** and **if-then extractor**. The Relation extractor extracts cause-effect relations from a sentence using a set of regular expressions. The if-then extractor extracts if-then relations from a sentence using a set of regular expressions. And it supports six languages: English, Italian, Spanish, French, Polish and Germany. English is the default language.No NLP tool is employed in both cases, so the functions are unaware of Part-Of-Speech or Dependency-Relations among words. And they are designed to be used with short texts from social media, and the input text string must be a single sentence and have already been pre-processed to remove non latin-1 characters like emojis and URLs.   ",
   version="1.0",
 contact={"name": "Tinsae Gebrechristos Dulecha; Carlo R. M. A. Santagiustina", "email": "tinsae.dulecha@unive.it; carlo.santagiustina@unive.it"}, redoc_url=None)
app = Flask(__name__)
myfastapi.mount("/home", WSGIMiddleware(app))
myfastapi.mount("/static", StaticFiles(directory="static"), name="static")

@myfastapi.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@myfastapi.get("/Causal_statement_extractor/")
async def Relation_extractor(sentence:str = Query(description="a character string containing a single sentence"),
                             consider_passive:bool = Query("true", description="consider passive form passive"),
                            consider_end_form:bool = Query("true", description="consider end of sentence verbs")
                           ):
    listres = relation_api(sentence, consider_passive, consider_end_form)
    return {"id": str(list(listres[0])[ 0]), "cause": str(list(listres[1])[0]), "rel_negation": str(list(listres[2])[0]),
            "rel_operator": str(list(listres[3])[ 0]), "rel_passive_form": str(list(listres[4])[0]), "rel_creation": str(list(listres[5])[0]),
            "rel_destruction": str(list(listres[6])[ 0]), "rel_causation": str(list(listres[7])[0]), "rel_coref_res": str(list(listres[8])[0]),
            "effect": str(list(listres[9])[0])}
@myfastapi.get("/if_then_extractor/")
async def IF_THEN_extrator(sentence:str=Query(description=" a character string containing a single sentence"), language:str = Query("en", description=" the language ISO 639-1 (2-character) code of the language to be used. Currently available languages: English=\"en\", Italian=\"it\", Polish=\"pl\", French=\"fr\", German=\"de\"  and Spanish=\"es\"")):
    ifpart, thenpart, recsentence =   if_then_api(sentence,language)
    return {"IF": ifpart, "THEN": thenpart, "sentence": recsentence}

@myfastapi.post("/Fileupload/", response_class=FileResponse )
async def if_then_extractor(in_file: UploadFile=File(...),
                        language:str = Query("en", description=" the language ISO 639-1 (2-character) code of the language to be used. Currently available languages: English=\"en\", Italian=\"it\", Polish=\"pl\", French=\"fr\", German=\"de\"  and Spanish=\"es\"")):
    out_file_path= os.path.join(app.config['TEMPAPIDATA_UPLOADFOLDER'], (in_file.filename))
    async with aiofiles.open(out_file_path, 'wb') as out_file:
        content = await in_file.read()  # async read
        await out_file.write(content)  # async write
    response = AE(out_file_path, language)
    os.remove(out_file_path)

    df = pd.DataFrame(response, columns=["If part", "Then part", "Full sentence"])
    return StreamingResponse(
        iter([df.to_csv(index=False)]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=extracted_file.csv"}
    )

@myfastapi.post("/Relation_extractor/", response_class=FileResponse )
async def relation_extractor(in_file: UploadFile=File(...),
                             consider_passive: bool = Query("true", description="consider passive form passive"),
                             consider_end_form: bool = Query("true", description="consider end of sentence verbs")
                             ):
    out_file_path= os.path.join(app.config['TEMPAPIDATA_UPLOADFOLDER'], (in_file.filename))
    async with aiofiles.open(out_file_path, 'wb') as out_file:
        content = await in_file.read()  # async read
        await out_file.write(content)  # async write
    response = Causal_AE(out_file_path, consider_passive, consider_end_form)
    os.remove(out_file_path)

    df = pd.DataFrame(response, columns=["Id", "Cause", "rel_negation","rel_operator", "rel_passive_form", "rel_creation","rel_destruction", "rel_causation", "rel_coref_res","effect"])
    return StreamingResponse(
        iter([df.to_csv(index=False)]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=extracted_file.csv"}
    )

app.secret_key = "dont tell anyone"
app.config['secret_key'] = "dont tell anyone"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMPAPIDATA_UPLOADFOLDER'] = TEMPAPIDATA_UPLOADFOLDER

ALLOWED_EXTENSIONS = set(['csv'])

def if_then_api(sentence,chlang):
    import rpy2.robjects as robjects
    r = robjects.r
    r['source']('if_then_extractor.R')
    if_then_extractor_sentence = robjects.globalenv['if_then_extractor_sentence']
    listres = if_then_extractor_sentence(sentence, chlang)
    ifpart = str(list(listres[1])[0])
    thenpart = str(list(listres[2])[0])
    return ifpart, thenpart, sentence

def relation_api(sentence,consider_passive, consider_end_form ):
    import rpy2.robjects as robjects
    r = robjects.r
    r['source']('if_then_extractor.R')
    relation_extractor = robjects.globalenv['relation_extractor']
    listres = relation_extractor(sentence, consider_passive, consider_end_form)

    return listres

def if_then_from_file(filename, lang):
    fname=filename
    os.remove(filename)
    return fname, lang
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# if then argument extrator #

def func(value):
    return ''.join(value.splitlines())

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)


def AE(filename, chlang):

    wholepart = []
    # import rpy2.robjects as robjects
    # r = robjects.r
    # r['source']('if_then_extractor.R')
    #
    # if_then_extractor_sentence = robjects.globalenv['if_then_extractor_sentence']
    total = 0
    exc = 0
    error_exc = []

    if filename.endswith('.csv'):
        from csv import reader

        # open file in read mode
        with open(filename, 'r', encoding='utf-8') as read_obj:
            # pass the file object to reader() to get the reader object
            csv_reader = reader(read_obj)
            # Iterate over each row in the csv using reader object
            for row in csv_reader:
                # row variable is a list that represents a row in csv
                text = row[1]
                text = emoji_pattern.sub(r'', text)
                sentenceS = sent_tokenize(text)
                for i in range(len(sentenceS)):

                    total += 1
                    fullsentence =sentenceS[i]

                    sentenceSi = re.sub(r'[\(\)]', '', fullsentence)
                    sentenceSi = sentenceSi.replace(" ,", ",")
                    if len(re.findall(r'\w+', sentenceSi)) > 4:

                        try:
                            ifpart, thenpart,_ = if_then_api(sentenceSi, chlang)
                        except:
                            exc += 1
                            error_exc.append([sentenceSi])
                            continue

                        # ifpart = str(list(listres[1])[0])
                        # thenpart = str(list(listres[2])[0])
                        # linecomplete = [line + ';' + sentenceSi + ';' + ifpart + ';' + thenpart][0]
                        if ifpart != '000000' and thenpart != "":
                            wholepart.append([ifpart, thenpart, fullsentence])

    elif filename.endswith('.json'):
        dk1 = "id"
        dk2 = "author_id"
        dk3 = 'in_reply_to_user_id'
        dk4 = "created_at"
        dk5 = "text"

        with open(filename, encoding='utf-8') as json_file:

            data = json.load(json_file)

            for dato in data:

                text = dato[dk5]
                text = func(text)
                text = emoji_pattern.sub(r'', text)
                # line = str(dato[dk1]) + ';' + str(dato[dk2]) + ';' + str(dato[dk4]) + ';' + str(text)

                sentenceS = sent_tokenize(text)
                for i in range(len(sentenceS)):

                    total += 1
                    fullsentence =sentenceS[i]
                    sentenceSi = re.sub(r'[\(\)]', '', fullsentence)
                    sentenceSi = sentenceSi.replace(" ,", ",")

                    try:
                        listres = if_then_extractor_sentence(sentenceSi, chlang)
                    except:
                        exc += 1
                        error_exc.append([sentenceSi])
                        continue

                    ifpart = str(list(listres[1])[0])
                    thenpart = str(list(listres[2])[0])
                    # linecomplete = [line + ';' + sentenceSi + ';' + ifpart + ';' + thenpart][0]
                    if ifpart != '000000' and thenpart != "":

                        wholepart.append([ifpart, thenpart, fullsentence])


    else:
        filetype = 'notsupported'

    return wholepart


def Causal_AE(filename, consider_passive, consider_end_form):
    wholepart = []
    cause_effect_full=[]
    total = 0
    exc = 0
    error_exc = []

    if filename.endswith('.csv'):
        from csv import reader

        # open file in read mode
        with open(filename, 'r', encoding='utf-8') as read_obj:
            # pass the file object to reader() to get the reader object
            csv_reader = reader(read_obj)
            next(csv_reader)

            # Iterate over each row in the csv using reader object
            for row in csv_reader:
                # row variable is a list that represents a row in csv
                text = row[1]
                if text!="":
                    text = emoji_pattern.sub(r'', text)
                    sentenceS = sent_tokenize(text)
                    for i in range(len(sentenceS)):

                        total += 1
                        fullsentence = sentenceS[i]

                        sentenceSi = re.sub(r'[\(\)]', '', fullsentence)
                        sentenceSi = sentenceSi.replace(" ,", ",")
                        if len(re.findall(r'\w+', sentenceSi)) > 4:

                            try:
                                listres = relation_api(sentenceSi, consider_passive, consider_end_form)
                                id = str(list(listres[0])[0]) if list(listres[0])[0] is not None else '000000'
                                cause = str(list(listres[1])[0]) if list(listres[1])[0] is not None else '000000'
                                rel_negation = str(list(listres[2])[0]) if list(listres[2])[0] is not None else '000000'
                                rel_operator = str(list(listres[3])[0]) if list(listres[3])[0] is not None else '000000'
                                rel_passive_form = str(list(listres[4])[0]) if list(listres[4])[
                                                                                   0] is not None else '000000'
                                rel_creation = str(list(listres[5])[0]) if list(listres[5])[0] is not None else '000000'
                                rel_destruction = str(list(listres[6])[0]) if list(listres[6])[
                                                                                  0] is not None else '000000'
                                rel_causation = str(list(listres[7])[0]) if list(listres[7])[
                                                                                0] is not None else '000000'
                                rel_coref_res = str(list(listres[8])[0]) if list(listres[8])[
                                                                                0] is not None else '000000'
                                effect = str(list(listres[9])[0]) if list(listres[9])[0] is not None else '000000'

                            except:
                                exc += 1
                                error_exc.append([sentenceSi])
                                continue


                            # linecomplete = [line + ';' + sentenceSi + ';' + ifpart + ';' + thenpart][0]
                            if cause != '000000' and effect != '000000':
                                wholepart.append(
                                    [id, cause, rel_negation, rel_operator, rel_passive_form, rel_creation,
                                     rel_destruction,
                                     rel_causation, rel_coref_res, effect])
                                cause_effect_full.append([cause, effect, fullsentence])

    elif filename.endswith('.json'):
        dk1 = "id"
        dk2 = "author_id"
        dk3 = 'in_reply_to_user_id'
        dk4 = "created_at"
        dk5 = "text"

        with open(filename, encoding='utf-8') as json_file:

            data = json.load(json_file)

            for dato in data:

                text = dato[dk5]
                text = func(text)
                text = emoji_pattern.sub(r'', text)
                # line = str(dato[dk1]) + ';' + str(dato[dk2]) + ';' + str(dato[dk4]) + ';' + str(text)

                sentenceS = sent_tokenize(text)
                for i in range(len(sentenceS)):

                    total += 1
                    fullsentence = sentenceS[i]

                    sentenceSi = re.sub(r'[\(\)]', '', fullsentence)
                    sentenceSi = sentenceSi.replace(" ,", ",")
                    if len(re.findall(r'\w+', sentenceSi)) > 4:

                        try:
                            listres = relation_api(sentenceSi, consider_passive, consider_end_form)
                            id = str(list(listres[0])[0]) if list(listres[0])[0] is not None else '000000'
                            cause = str(list(listres[1])[0]) if list(listres[1])[0] is not None else '000000'
                            rel_negation = str(list(listres[2])[0]) if list(listres[2])[0] is not None else '000000'
                            rel_operator = str(list(listres[3])[0]) if list(listres[3])[0] is not None else '000000'
                            rel_passive_form = str(list(listres[4])[0]) if list(listres[4])[
                                                                               0] is not None else '000000'
                            rel_creation = str(list(listres[5])[0]) if list(listres[5])[0] is not None else '000000'
                            rel_destruction = str(list(listres[6])[0]) if list(listres[6])[
                                                                              0] is not None else '000000'
                            rel_causation = str(list(listres[7])[0]) if list(listres[7])[
                                                                            0] is not None else '000000'
                            rel_coref_res = str(list(listres[8])[0]) if list(listres[8])[
                                                                            0] is not None else '000000'
                            effect = str(list(listres[9])[0]) if list(listres[9])[0] is not None else '000000'

                        except:
                            exc += 1
                            error_exc.append([sentenceSi])
                            continue

                        # linecomplete = [line + ';' + sentenceSi + ';' + ifpart + ';' + thenpart][0]
                        if cause != '000000' and effect != '000000':
                            wholepart.append(
                                [id, cause, rel_negation, rel_operator, rel_passive_form, rel_creation,
                                 rel_destruction,
                                 rel_causation, rel_coref_res, effect])
                            cause_effect_full.append([cause, effect, fullsentence])


    else:
        flash('Unsupported file type. Please upload csv file')
    return wholepart, cause_effect_full

# 2.function to get string from list.
def get_string(lst, language):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import nltk
    import pandas as pd
    from stop_words import get_stop_words
    stop_words = get_stop_words(language)
    allwords = []
    df_format = pd.DataFrame()
    for i in range(len(lst)):
        sentence = lst[i]
        sentence = sentence.lower()
        sentence = re.sub(r'[^\w\s]', ' ', sentence)
        sentence = re.sub(r'[0-9]', ' ', sentence)

        sentence = re.sub('@[^\s]+', ' ', sentence)
        sentence = re.sub(' amp ', ' ', sentence)
        sentence = re.sub('http[^\s]+', ' ', sentence)
        sentence = re.sub(r'http\S+', '', sentence, flags=re.MULTILINE)

        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        filtered_thenstmt = word_tokenize(remove_stopwords(sentence, stopwords=stop_words))
        if filtered_thenstmt:
            allwords.append(filtered_thenstmt)
        words = nltk.RegexpTokenizer(r'\w+').tokenize(str(allwords))

        for word in words:
            if len(word)<3:
                words.remove(word)




        #words=get_word(str(allwords))
        Freq_dist_nltk = nltk.FreqDist(words)
        df_freq = pd.DataFrame.from_dict(Freq_dist_nltk, orient='index')
        df_freq.columns = ['Frequency']
        df_freq.index.name = 'Term'
        df_freq = df_freq.sort_values(by=['Frequency'],ascending=False)
        df_freq = df_freq.reset_index()
        df_format=df_freq
        df_freq=df_freq.values.tolist()
       #freqdist = Freq_df(words)

    return df_freq, words, df_format



@app.route("/datasets", methods=['POST', 'GET'])
def datasets(lang=None):
    filenames = os.listdir('static/uploads/')

    return render_template("datasets.html", filenames=filenames, lang=lang)


@app.route("/example_dataset")
def example_dataset():
  with open("example.csv", 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    return render_template("example_dataset.html", csv=reader)
@app.route("/userguide")
def userguide():
    return render_template("userguide.html")
@app.route("/demovideo")
def demovideo():
    return render_template("userguide.html")
@app.route("/example_pipeline", methods=['POST', 'GET'])
def example_pipeline():
    session['lang'] = "en"

    session['uploaded_data_file_path'] = "static/example/en_example.csv"

    session['graphJSON'] = session['graphJSON'] if session.get('graphJSON') else ""
    session['filepath'] = session['filepath'] if session.get('filepath') else ""
    session['filename'] = session['filename'] if session.get('filename') else ""
    if request.method == 'POST':

                if request.form['example_button'] == 'Extract Argument':
                    session['filepath']=""
                    session['graphJSON']=""
                    session['filename']=""
                    if_then_full = AE((session['uploaded_data_file_path']), "en")

                    if_then_array = np.array(if_then_full)

                    allthenpart = if_then_array[:, 1]
                    allthenpart = allthenpart.tolist()

                    allfsentences = if_then_array[:, 2]
                    allfsentences = allfsentences.tolist()

                    session['procfile'] = allthenpart
                    session['fln'] = allfsentences
                    session['fullinfo'] = if_then_full
                    session['freqdist'] = ""
                    session['words'] = ""
                    session['df_format'] = ""

                    return render_template('example.html', invisibiltystep2='visible', flines=session['fln'], fullfile=session['fullinfo'])

                elif request.form['example_button'] == 'Wordcloud':
                    if ((session.get('freqdist') == "") or (session.get('words') == "")):
                        freqdist, words, df_format = get_string(session['procfile'], session['lang'])
                        session['freqdist'] = freqdist
                        session['words'] = words
                        session['df_format'] = df_format
                    imagenumber = str(random.randint(0, 100000))
                    unique_string = str(list(session['words']))
                    wordcloud = WordCloud(collocations=True, background_color="black", min_font_size=4).generate(
                        unique_string)
                    wordcloud.to_file('static/wordcloud' + imagenumber + '.png')
                    session['filename'] = "wordcloud" + imagenumber + ".png"

                    return render_template('example.html', invisibiltystep3='visible', flines=session['fln'],  argument = session['procfile'], datafr=session['df_format'],
                                           plot_wordcloud='visible', fullfile=session['fullinfo'],  graphJSON=session['graphJSON'],fpath=session['filepath'])
                elif request.form['example_button'] == 'Graph':
                    return render_template('example.html', invisibiltystep2='visible', filepath=session['uploaded_data_file_path'], plotgraph='visible')

                elif request.form['example_button'] == 'Plot Wordcloud':
                    session['filepath'] = "static/" + session['filename']
                    session['graphJSON'] = session['graphJSON'] if session.get('graphJSON') else ""
                    session['filename'] = session['filename'] if session.get('filename') else ""
                    return render_template('example.html', invisibiltystep3='visible', flines=session['fln'],plot_wordcloud='visible', argument = session['procfile'],
                                           datafr=session['df_format'],wordcloudimg=session['filename'] , graphJSON=session['graphJSON'],fpath=session['filepath'],fullfile=session['fullinfo'])


                elif request.form['example_button'] == 'Word Frequency':
                    if ((session.get('freqdist') == "") or (session.get('words') == "")):
                        freqdist, words, df_format = get_string(session['procfile'], session['lang'])
                        session['freqdist'] = freqdist
                        session['words'] = words
                        session['df_format'] = df_format


                    return render_template('example.html', invisibiltystep3='visible', flines=session['fln'],  argument = session['procfile'],fullfile=session['fullinfo'],
                                           plot_wordfrequency='visible', wordcloudimg=session['filename'] , graphJSON=session['graphJSON'],fpath=session['filepath'])
                elif request.form['example_button'] == 'Plot Word frequency':

                    #fig = px.histogram(session[df_format], x='continent', y='col_chosen', histfunc='avg')
                    fig = px.scatter(session['df_format'].query(" Frequency > 2"), x='Term', y='Frequency',
                                     hover_data=['Term', 'Frequency'], size= 'Frequency', size_max=40,
                                     color='Frequency')

                    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                    session['graphJSON']=graphJSON
                    session['filepath'] = session['filepath'] if session.get('filepath') else ""
                    session['filename'] = session['filename'] if session.get('filename') else ""


                    return render_template('example.html', graphJSON=graphJSON, invisibiltystep3='visible', flines=session['fln'],   fqdist = session['freqdist'],
                                           datafr=session['df_format'], wordcloudimg=session['filename'], fpath=session['filepath'],fullfile=session['fullinfo'],plot_wordfrequency='visible')

    else:
        return render_template('example.html', filename=session['uploaded_data_file_path'])


# causal example pipeline
@app.route("/cexample_pipeline", methods=['POST', 'GET'])
def cexample_pipeline():

    session['lang'] = "en"

    session['uploaded_data_file_path'] = "static/example/testcausal.csv"

    if request.method == 'POST':
        if request.form['example_button'] == 'Extract Argument':
            session['filepath']=""
            session['graphJSON']=""
            session['filename']=""
            _, if_then_full = Causal_AE(session['uploaded_data_file_path'], True, True)
            if_then_array = np.array(if_then_full)

            cause = if_then_array[:, 0]
            cause = cause.tolist()
            effect = if_then_array[:, 1]
            effect = effect.tolist()

            allfsentences = if_then_array[:, 2]
            allfsentences = allfsentences.tolist()

            session['procfile'] = effect
            session['fln'] = allfsentences
            session['fullinfo'] = if_then_full
            session['freqdist'] = ""
            session['words'] = ""
            session['df_format'] = ""

            return render_template('causal_example.html', invisibiltystep2='visible', flines=session['fln'], fullfile=session['fullinfo'])

        elif request.form['example_button'] == 'Wordcloud':
            if ((session.get('freqdist') == "") or (session.get('words') == "")):
                freqdist, words, df_format = get_string(session['procfile'], session['lang'])
                session['freqdist'] = freqdist
                session['words'] = words
                session['df_format'] = df_format
            imagenumber = str(random.randint(0, 100000))
            unique_string = str(list(session['words']))
            wordcloud = WordCloud(collocations=True, background_color="black", min_font_size=4).generate(
                unique_string)
            wordcloud.to_file('static/wordcloud' + imagenumber + '.png')
            session['filename'] = "wordcloud" + imagenumber + ".png"
            return render_template('causal_example.html', invisibiltystep3='visible', flines=session['fln'],  argument = session['procfile'], datafr=session['df_format'],
                                   plot_wordcloud='visible', fullfile=session['fullinfo'],  graphJSON=session['graphJSON'],fpath=session['filepath'])
        elif request.form['example_button'] == 'Graph':
            return render_template('causal_example.html', invisibiltystep2='visible', filepath=session['uploaded_data_file_path'], plotgraph='visible')

        elif request.form['example_button'] == 'Plot Wordcloud':
            session['filepath'] = "static/" + session['filename']
            session['graphJSON'] = session['graphJSON'] if session.get('graphJSON') else ""
            session['filename'] = session['filename'] if session.get('filename') else ""
            return render_template('causal_example.html', invisibiltystep3='visible', flines=session['fln'],plot_wordcloud='visible', argument = session['procfile'],
                                   datafr=session['df_format'],wordcloudimg=session['filename'] , graphJSON=session['graphJSON'],fpath=session['filepath'],fullfile=session['fullinfo'])


        elif request.form['example_button'] == 'Word Frequency':
            if ((session.get('freqdist') == "") or (session.get('words') == "")):
                freqdist, words, df_format = get_string(session['procfile'], session['lang'])
                session['freqdist'] = freqdist
                session['words'] = words
                session['df_format'] = df_format


            return render_template('causal_example.html', invisibiltystep3='visible', flines=session['fln'],  argument = session['procfile'],fullfile=session['fullinfo'],
                                   plot_wordfrequency='visible', wordcloudimg=session['filename'] , graphJSON=session['graphJSON'],fpath=session['filepath'])
        elif request.form['example_button'] == 'Plot Word frequency':

            #fig = px.histogram(session[df_format], x='continent', y='col_chosen', histfunc='avg')
            fig = px.scatter(session['df_format'].query(" Frequency > 2"), x='Term', y='Frequency',
                             hover_data=['Term', 'Frequency'], size= 'Frequency', size_max=40,
                             color='Frequency')

            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            session['graphJSON']=graphJSON
            session['filepath'] = session['filepath'] if session.get('filepath') else ""
            session['filename'] = session['filename'] if session.get('filename') else ""


            return render_template('causal_example.html', graphJSON=graphJSON, invisibiltystep3='visible', flines=session['fln'],   fqdist = session['freqdist'],
                                   datafr=session['df_format'], wordcloudimg=session['filename'], fpath=session['filepath'],fullfile=session['fullinfo'],plot_wordfrequency='visible')

    else:
        return render_template('causal_example.html', filename=session['uploaded_data_file_path'])

@app.route("/download/<fname>", methods=['POST', 'GET'])
def downloadfile(fname):
    fpath="static/uploads/" + fname
    return send_file(fpath, as_attachment=True)
@app.route("/export/<fname>", methods=['POST', 'GET'])
def exportfile(fname):
    return send_file(fname, as_attachment=True)

@app.route("/deletefile/<fname>", methods=['POST', 'GET'])
def deletefile(fname):
    fpath="static/uploads/" + fname

    os.remove(fpath)
    return redirect(url_for('datasets'))

@app.route("/downloadim/<imagename>", methods=['POST', 'GET'])
def downloadimage(imagename):
    imagename="static/" + imagename
    return send_file(imagename, as_attachment=True)



@app.route("/analyzefile/<fname>", methods=['POST', 'GET'])
def analyzefile(fname):
    session['procfile'] = ""
    session['fln'] = ""
    splitted_text=fname.split('_')
    lang= splitted_text[0]
    data_filename = "static/uploads/" + fname
    session['lang'] = lang
    session['uploaded_data_file_path'] = data_filename
    if_then_full = AE((session['uploaded_data_file_path']),lang)

    if_then_array = np.array(if_then_full)

    allthenpart = if_then_array[:, 1]
    allthenpart = allthenpart.tolist()

    allfsentences = if_then_array[:, 2]
    allfsentences = allfsentences.tolist()

    session['procfile'] = allthenpart
    session['fln'] = allfsentences
    session['fullinfo'] = if_then_full

    freqdist, words, df_format = get_string(session['procfile'], session['lang'])
    session['freqdist'] = freqdist
    session['words'] = words
    session['df_format'] = df_format
    session['graphJSON'] = session['graphJSON'] if session.get('graphJSON') else ""
    session['filepath'] = session['filepath'] if session.get('filepath') else ""
    session['filename'] = session['filename'] if session.get('filename') else ""
    if request.method == 'POST':

                if request.form['example_button'] == 'Extract Argument':
                    session['filepath']=""
                    session['graphJSON']=""
                    session['filename']=""
                    return render_template('analyze.html', invisibiltystep2='visible', flines=session['fln'],
                                           fullfile=session['fullinfo'])
                elif request.form['example_button'] == 'Wordcloud':
                    imagenumber = str(random.randint(0, 100000))

                    unique_string = str(list(session['words']))
                    wordcloud = WordCloud(collocations=True, background_color="black", min_font_size=4).generate(
                        unique_string)

                    wordcloud.to_file('static/wordcloud' + imagenumber + '.png')
                    session['filename'] = "wordcloud" + imagenumber + ".png"

                    return render_template('analyze.html', invisibiltystep3='visible', flines=session['fln'],
                                           argument=session['procfile'], datafr=session['df_format'],
                                           plot_wordcloud='visible', fullfile=session['fullinfo'],
                                         graphJSON=session['graphJSON'],
                                           fpath=session['filepath'])
                elif request.form['example_button'] == 'Graph':
                    return render_template('analyze.html', invisibiltystep2='visible',
                                           filepath=session['uploaded_data_file_path'], plotgraph='visible')

                elif request.form['example_button'] == 'Plot Wordcloud':
                    session['filepath'] = "static/" + session['filename']
                    session['graphJSON'] = session['graphJSON'] if session.get('graphJSON') else ""
                    session['filename'] = session['filename'] if session.get('filename') else ""
                    return render_template('analyze.html', invisibiltystep3='visible', flines=session['fln'],
                                           plot_wordcloud='visible', argument=session['procfile'],
                                           datafr=session['df_format'], wordcloudimg=session['filename'],
                                           graphJSON=session['graphJSON'], fpath=session['filepath'],
                                           fullfile=session['fullinfo'])


                elif request.form['example_button'] == 'Word Frequency':

                    return render_template('analyze.html', invisibiltystep3='visible', flines=session['fln'],
                                           argument=session['procfile'], fullfile=session['fullinfo'],
                                           plot_wordfrequency='visible', wordcloudimg=session['filename'],
                                           graphJSON=session['graphJSON'], fpath=session['filepath'])
                elif request.form['example_button'] == 'Plot Word frequency':

                    # fig = px.histogram(session[df_format], x='continent', y='col_chosen', histfunc='avg')
                    fig = px.scatter(session['df_format'].query(" Frequency > 2"), x='Term', y='Frequency',
                                     hover_data=['Term', 'Frequency'], size='Frequency', size_max=40,
                                     color='Frequency')

                    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                    session['graphJSON'] = graphJSON
                    session['filepath'] = session['filepath'] if session.get('filepath') else ""
                    session['filename'] = session['filename'] if session.get('filename') else ""

                    return render_template('analyze.html', graphJSON=graphJSON, invisibiltystep3='visible',
                                           flines=session['fln'], fqdist=session['freqdist'],
                                           datafr=session['df_format'], wordcloudimg=session['filename'],
                                           fpath=session['filepath'], fullfile=session['fullinfo'],
                                           plot_wordfrequency='visible')

    else:
        return render_template('analyze.html', filename=session['fln'])



@app.route('/pipeline', methods=['POST', 'GET'])
def pipeline():
    session['flines'] = ""

    if request.method == 'POST':
           f = request.files.get('filename')
        # Extracting uploaded file name
           data_filename = secure_filename(f.filename)
           lang = request.form['langmenu']
           session['lang'] = lang


           if data_filename != '':
                    f.save(os.path.join(app.config['UPLOAD_FOLDER'], (lang + "_" + data_filename)))
                    session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], (lang + "_" + data_filename))

                    return render_template('pipeline.html', filename= session['uploaded_data_file_path'])
           else:
                if request.form['submit_button'] == 'Extract Argument':
                    session['filepath']=""
                    session['graphJSON']=""
                    session['filename']=""
                    head_tail = os.path.split(session['uploaded_data_file_path'])

                    filename=head_tail[1]
                    session['lang'] = filename.split('_')[0]

                    if_then_full = AE((session['uploaded_data_file_path']),session['lang'])

                    if_then_array = np.array(if_then_full)

                    allthenpart = if_then_array[:, 1]
                    allthenpart = allthenpart.tolist()

                    allfsentences = if_then_array[:, 2]
                    allfsentences = allfsentences.tolist()

                    session['processedfile'] = allthenpart
                    session['fline'] = allfsentences
                    session['fullinfo'] = if_then_full
                    session['uploaded_file'] = data_filename
                    session['freqdist'] = ""
                    session['words'] = ""
                    session['df_format'] = ""

                    return render_template('pipeline.html', invisibiltystep2='visible', flines=session['fline'],
                                           fullfile=session['fullinfo'])

                elif request.form['submit_button'] == 'Wordcloud':
                    if ((session.get('freqdist') == "") or (session.get('words') == "")):
                        freqdist, words, df_format = get_string(session['processedfile'], session['lang'])
                        session['freqdist'] = freqdist
                        session['words'] = words
                        session['df_format'] = df_format
                    unique_string = str(list(session['words']))
                    wordcloud = WordCloud(collocations=True, background_color="black", min_font_size=4).generate(
                        unique_string)
                    imagenumber = str(random.randint(0, 100000))

                    wordcloud.to_file('static/wordcloud' + imagenumber + '.png')
                    session['filename'] = "wordcloud" + imagenumber + ".png"

                    return render_template('pipeline.html', invisibiltystep3='visible', flines=session['fline'],  argument = session['processedfile'], datafr=session['df_format'],
                                           plot_wordcloud='visible', fullfile=session['fullinfo'], graphJSON=session['graphJSON'],
                                           fpath=session['filepath'])


                elif request.form['submit_button'] == 'Graph':
                    return render_template('pipeline.html', invisibiltystep2='visible', filepath=session['uploaded_data_file_path'])
                elif request.form['submit_button'] == 'Plot Wordcloud':
                    session['filepath'] = "static/" + session['filename']
                    session['graphJSON'] = session['graphJSON'] if session.get('graphJSON') else ""
                    session['filename'] = session['filename'] if session.get('filename') else ""
                    return render_template('pipeline.html', invisibiltystep3='visible', flines=session['fline'],
                                           plot_wordcloud='visible', argument=session['processedfile'],
                                           datafr=session['df_format'], wordcloudimg=session['filename'],
                                           graphJSON=session['graphJSON'], fpath=session['filepath'],
                                           fullfile=session['fullinfo'])

                elif request.form['submit_button'] == 'Word Frequency':
                    if ((session.get('freqdist') == "") or (session.get('words') == "")):
                        freqdist, words, df_format = get_string(session['processedfile'], session['lang'])
                        session['freqdist'] = freqdist
                        session['words'] = words
                        session['df_format'] = df_format
                    return render_template('pipeline.html', invisibiltystep3='visible', flines=session['fline'],
                                           argument=session['processedfile'], fullfile=session['fullinfo'],
                                           plot_wordfrequency='visible', wordcloudimg=session['filename'],
                                           graphJSON=session['graphJSON'], fpath=session['filepath'])
                elif request.form['submit_button'] == 'Plot Word frequency':

                    # fig = px.histogram(session[df_format], x='continent', y='col_chosen', histfunc='avg')
                    fig = px.scatter(session['df_format'].query(" Frequency > 2"), x='Term', y='Frequency',
                                     hover_data=['Term', 'Frequency'], size='Frequency', size_max=40,
                                     color='Frequency')

                    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                    session['graphJSON'] = graphJSON
                    session['filepath'] = session['filepath'] if session.get('filepath') else ""
                    session['filename'] = session['filename'] if session.get('filename') else ""

                    return render_template('pipeline.html', graphJSON=graphJSON, invisibiltystep3='visible',
                                           flines=session['fline'], fqdist=session['freqdist'],
                                           datafr=session['df_format'], wordcloudimg=session['filename'],
                                           fpath=session['filepath'], fullfile=session['fullinfo'],
                                           plot_wordfrequency='visible')

    else:
            return render_template("pipeline.html")


## causal pipeline

@app.route('/causal_pipeline', methods=['POST', 'GET'])
def causal_pipeline():
    session['processedfile'] = ""
    session['flines'] = ""
    if request.method == 'POST':
        f = request.files.get('filename')
        # Extracting uploaded file name
        data_filename = secure_filename(f.filename)
        if data_filename != '':
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
            session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
            session['procfile'] = ""
            session['fln'] = ""
            session['lang'] = "en"

            return render_template('causal_pipeline.html', filename=session['uploaded_data_file_path'],)
        else:
            if request.form['submit_button'] == 'Extract Argument':
                session['filepath'] = ""
                session['graphJSON'] = ""
                session['filename'] = ""
                _, if_then_full = Causal_AE(session['uploaded_data_file_path'], True, True)
                if_then_array = np.array(if_then_full)

                cause = if_then_array[:, 0]
                cause = cause.tolist()
                effect = if_then_array[:, 1]
                effect = effect.tolist()

                allfsentences = if_then_array[:, 2]
                allfsentences = allfsentences.tolist()

                session['procfile'] = effect
                session['fln'] = allfsentences
                session['fullinfo'] = if_then_full
                session['freqdist'] = ""
                session['words'] = ""
                session['df_format'] = ""

                return render_template('causal_pipeline.html', invisibiltystep2='visible', flines=session['fln'],
                                       fullfile=session['fullinfo'])

            elif request.form['submit_button'] == 'Wordcloud':
                if ((session.get('freqdist')=="") or (session.get('words')=="")):
                      freqdist, words, df_format = get_string(session['procfile'], session['lang'])
                      session['freqdist'] = freqdist
                      session['words'] = words
                      session['df_format'] = df_format


                session['graphJSON'] = session['graphJSON'] if session.get('graphJSON') else ""
                session['filepath'] = session['filepath'] if session.get('filepath') else ""
                session['filename'] = session['filename'] if session.get('filename') else ""
                unique_string = str(list(session['words']))
                wordcloud = WordCloud(collocations=True, background_color="black", min_font_size=4).generate(
                    unique_string)
                imagenumber = str(random.randint(0, 100000))

                wordcloud.to_file('static/wordcloud' + imagenumber + '.png')
                session['filename'] = "wordcloud" + imagenumber + ".png"

                return render_template('causal_pipeline.html', invisibiltystep3='visible', flines=session['fln'],
                                       argument=session['processedfile'], datafr=session['df_format'],
                                       plot_wordcloud='visible', fullfile=session['fullinfo'],
                                       graphJSON=session['graphJSON'],
                                       fpath=session['filepath'])


            elif request.form['submit_button'] == 'Graph':
                return render_template('causal_pipeline.html', invisibiltystep2='visible',
                                       filepath=session['uploaded_data_file_path'])
            elif request.form['submit_button'] == 'Plot Wordcloud':
                session['filepath'] = "static/" + session['filename']
                session['graphJSON'] = session['graphJSON'] if session.get('graphJSON') else ""
                session['filename'] = session['filename'] if session.get('filename') else ""
                return render_template('causal_pipeline.html', invisibiltystep3='visible', flines=session['fln'],
                                       plot_wordcloud='visible', argument=session['processedfile'],
                                       datafr=session['df_format'], wordcloudimg=session['filename'],
                                       graphJSON=session['graphJSON'], fpath=session['filepath'],
                                       fullfile=session['fullinfo'])

            elif request.form['submit_button'] == 'Word Frequency':
                if ((session.get('freqdist') == "") or (session.get('words') == "")):
                    freqdist, words, df_format = get_string(session['procfile'], session['lang'])
                    session['freqdist'] = freqdist
                    session['words'] = words
                    session['df_format'] = df_format
                session['graphJSON'] = session['graphJSON'] if session.get('graphJSON') else ""
                session['filepath'] = session['filepath'] if session.get('filepath') else ""
                session['filename'] = session['filename'] if session.get('filename') else ""
                return render_template('causal_pipeline.html', invisibiltystep3='visible', flines=session['fln'],
                                       argument=session['processedfile'], fullfile=session['fullinfo'],
                                       plot_wordfrequency='visible', wordcloudimg=session['filename'],
                                       graphJSON=session['graphJSON'], fpath=session['filepath'])
            elif request.form['submit_button'] == 'Plot Word frequency':

                # fig = px.histogram(session[df_format], x='continent', y='col_chosen', histfunc='avg')
                fig = px.scatter(session['df_format'].query(" Frequency > 2"), x='Term', y='Frequency',
                                 hover_data=['Term', 'Frequency'], size='Frequency', size_max=40,
                                 color='Frequency')

                graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                session['graphJSON'] = graphJSON
                session['filepath'] = session['filepath'] if session.get('filepath') else ""
                session['filename'] = session['filename'] if session.get('filename') else ""

                return render_template('causal_pipeline.html', graphJSON=graphJSON, invisibiltystep3='visible',
                                       flines=session['fln'], fqdist=session['freqdist'],
                                       datafr=session['df_format'], wordcloudimg=session['filename'],
                                       fpath=session['filepath'], fullfile=session['fullinfo'],
                                       plot_wordfrequency='visible')

    else:
        return render_template("causal_pipeline.html")


if __name__ == "__main__":
    uvicorn.run(myfastapi, host='127.0.0.1', port=8000)

""" if __name__ == "__main__":
    app.run(debug=True)
 """






