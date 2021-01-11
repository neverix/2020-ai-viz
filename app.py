import json
from datetime import datetime
import streamlit as st
from plotly import express as px
import pandas as pd
from collections import Counter
import nltk
nltk.download("stopwords")
nltk.download("punkt")
import pymystem3
from string import punctuation
import wordcloud
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np


@st.cache
def read_file(fn):
    source = json.load(fn)
    result = []
    for message in source["messages"]:
        if message["type"] != "message":
            continue
        text = message["text"]
        if isinstance(text, list):
            text = ''.join([el if isinstance(el, str) else el["text"] for el in text])
        date = datetime.strptime(message["date"], "%Y-%m-%dT%H:%M:%S")
        if text:
            result.append((date, message["from"], text))
    return result


def hist(x, x_label, y_label="Число сообщений", **kwargs):
    df = pd.DataFrame(x, columns=[x_label])
    h = px.histogram(df, x=x_label, **kwargs)
    h.layout.yaxis.title.text = y_label
    return h


def bar(x, y, x_label, y_label="Число сообщений", limit=999, **kwargs):
    df = pd.DataFrame(sorted(zip(y, x), reverse=True)[:limit], columns=[y_label, x_label])
    return px.bar(df, x=x_label, y=y_label)


def scatter(x, y, x_label, y_label):
    return px.scatter(pd.DataFrame(zip(x, y), columns=[x_label, y_label]), x=x_label, y=y_label)


@st.cache
def split_words(messages):
    return [word for message in messages for word in nltk.word_tokenize(message, language="russian")]


@st.cache
def prep_words(words):
    lemmer = pymystem3.Mystem()
    stopwords = set(nltk.corpus.stopwords.words("russian") + ["весь", "это"])
    tokens = lemmer.lemmatize(' '.join(words).lower())
    tokens = [token.replace(' ', '') for token in tokens]
    tokens = [token for token in tokens if token
              and token not in stopwords
              and not all(char in punctuation or char.isnumeric() for char in token)]
    return tokens


def color_func(word=None, font_size=None,
               position=None, orientation=None,
               font_path=None, random_state=None):
    return f"hsl({random_state.randint(230, 270)}, {110}%, {60}%)"


@st.cache
def gen_wc(words):
    return wordcloud.WordCloud(background_color="white", max_font_size=80, random_state=0, width=800, height=480,
                               mask=np.array(Image.open("res/brain.jpg")), color_func=color_func,
                               font_path="res/RobotoCondensed-Regular.ttf") \
        .generate_from_frequencies({k: v for k, v in words.items() if v > 35})


@st.cache
def compute_stats(file):
    dts, authors, messages = map(list, zip(*file))
    author_counts = Counter(authors)
    counts, names = zip(*sorted(zip(author_counts.values(), author_counts.keys()), reverse=True))
    lens = [[len(msg) for msg, author in zip(messages, authors) if author == name] for name in names]
    totals = [sum(l) for l in lens]
    lens = [sum(l) / len(l) for l in lens]
    words = Counter(prep_words(split_words(messages)))
    worded = sorted(zip(words.values(), words.keys()), reverse=True)
    return dts, authors, messages, author_counts, counts, names, lens, totals, words, worded


def main():
    st.header("Анализ чата")
    file = read_file(open("res/result.json"))
    filename = st.file_uploader("Загрузить файл", type="json")
    if filename:
        file = read_file(filename)
    dts, authors, messages, author_counts, counts, names, lens, totals, words, worded = compute_stats(file)

    with st.beta_expander("Распределение сообщений"):
        # open("messages.txt", 'w').write('\n'.join(author + ': ' + line.replace('\n', '\\n')
        #                                 for author, line in zip(authors, messages)))

        st.subheader("По дням")
        st.plotly_chart(hist(dts, "День"))
        st.subheader("По часам")
        hours = [dt.hour for dt in dts]
        st.plotly_chart(hist(hours, "Час"))

        st.subheader("По длине")
        st.plotly_chart(hist(lens, "Длина"))
        st.subheader("По автору")
        st.plotly_chart(hist([names.index(author) for author in authors], "Авторы (пока анонимно)"))

        st.subheader("Средняя длина сообщений")
        # st.plotly_chart(scatter(counts, lens, "Количество сообщений", "Средняя длина"))
        lens_ = lens[:50]
        st.plotly_chart(scatter(range(1, len(lens_)+1), lens_, "Место", "Средняя длина"))

        st.subheader("Общее число символов")
        totals_ = totals[:50]
        st.plotly_chart(scatter(range(1, len(totals_)+1), totals_, "Место", "Число символов"))

    with st.beta_expander("Популярные слова"):
        st.subheader("Распределение")
        st.plotly_chart(bar(words.keys(), words.values(), "Слово", limit=250))

        st.subheader("Wordcloud")
        wc = gen_wc(words)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        st.subheader("Таблица")
        rows, cols = 25, 3
        columns = []
        for i in range(cols):
            columns.append(pd.DataFrame(worded[i:rows*cols:cols], columns=[f"Частота", f"Слово"]))
        table = pd.concat(columns, axis=1)
        st.table(table.assign(hack='').set_index('hack'))

    st.header("Лидерборды")

    def leaderboard(counts, name):
        # st.subheader(name)
        with st.beta_expander(name):
            st.plotly_chart(bar(names, counts, "Автор", name, limit=35))
            leaderboard = pd.DataFrame(sorted(zip(counts, names), reverse=True),
                                       columns=[name, "Автор"])
            leaderboard.index += 1
            leaderboard.index.name = "Место"
            st.table(leaderboard.head(100))

    leaderboard(counts, "Число сообщений")
    leaderboard(lens, "Средняя длина")
    leaderboard(totals, "Число символов")


if __name__ == '__main__':
    main()
