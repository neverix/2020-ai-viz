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


def main():
    st.header("Анализ чата")
    file = read_file(open("res/result.json"))
    filename = st.file_uploader("Загрузить файл", type="json")
    if filename:
        file = read_file(filename)

    st.header("Распределение сообщений")
    dts, authors, messages = map(list, zip(*file))

    open("messages.txt", 'w').write('\n'.join(author + ': ' + line.replace('\n', '\\n')
                                    for author, line in zip(authors, messages)))

    st.subheader("По дням")
    st.plotly_chart(hist(dts, "День"))
    st.subheader("По часам")
    hours = [dt.hour for dt in dts]
    st.plotly_chart(hist(hours, "Час"))

    st.subheader("По длине")
    lens = [len(msg) for msg in messages]
    st.plotly_chart(hist(lens, "Длина"))
    st.subheader("По автору")
    author_counts = Counter(authors)
    author_toks = [x[1] for x in sorted(zip(author_counts.values(), author_counts.keys()), reverse=True)]
    st.plotly_chart(hist([author_toks.index(author) for author in authors], "Авторы (пока анонимно)"))

    st.subheader("Средняя длина сообщений")
    author_counts = Counter(authors)
    counts, names = zip(*sorted(zip(author_counts.values(), author_counts.keys()), reverse=True)[:50])
    lens = [[len(msg) for msg, author in zip(messages, authors) if author == name] for name in names]
    lens = [sum(l) / len(l) for l in lens]
    # st.plotly_chart(scatter(counts, lens, "Количество сообщений", "Средняя длина"))
    st.plotly_chart(scatter(range(1, len(lens)+1), lens, "Место", "Средняя длина"))

    st.header("Популярные слова")

    st.subheader("Распределение")
    words = Counter(prep_words(split_words(messages)))
    st.plotly_chart(hist(words.values(), "Слово"))

    st.subheader("Таблица")
    worded = sorted(zip(words.values(), words.keys()), reverse=True)
    rows, cols = 10, 5
    columns = []
    for i in range(cols):
        columns.append(pd.DataFrame(worded[i:rows*cols:cols], columns=[f"Частота", f"Слово"]))
    table = pd.concat(columns, axis=1)
    st.table(table.assign(hack='').set_index('hack'))

    st.subheader("Wordcloud")
    def color_func(word=None, font_size=None,
                   position=None, orientation=None,
                   font_path=None, random_state=None):
        return f"hsl({random_state.randint(230, 270)}, {110}%, {60}%)"
    wc = wordcloud.WordCloud(background_color="white", max_font_size=80, random_state=0, width=800, height=480,
                             mask=np.array(Image.open("res/brain.jpg")), color_func=color_func,
                             font_path="res/RobotoCondensed-Regular.ttf")\
        .generate_from_frequencies({k: v for k, v in words.items() if v > 35})
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    st.header("Гиганты мысли")
    st.subheader("График")
    author_toks = [val for count, val in
                   sorted(zip(author_counts.values(), author_counts.keys()), reverse=True)[:35]
                   for _ in range(count)]
    st.plotly_chart(hist(author_toks, "Автор"))

    st.subheader("Лидерборд")
    leaderboard = pd.DataFrame(sorted(zip(author_counts.values(), author_counts.keys()), reverse=True),
                               columns=["Число сообщений", "Автор"])
    leaderboard.index += 1
    leaderboard.index.name = "Место"
    st.table(leaderboard.head(100))


if __name__ == '__main__':
    main()
