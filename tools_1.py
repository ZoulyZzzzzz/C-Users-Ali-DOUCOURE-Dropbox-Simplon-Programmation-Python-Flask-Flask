import pandas as pd
import re
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud


"""
1) Charger un csv

input : path_file
output: DF
"""

def load_csv(path_file):
    return pd.read_csv(path_file)


"""
2) Macro nettoyage + definition corpus

input: DF
output: deux corpus (positif et negatif)
"""


def macro_net(df):
    # Enlever les duplicats
    df.drop_duplicates(subset=['Texte'], inplace=True)
    # Creer un colonne longueur texte qui renvoi le nombre de mot par texte
    long = []
    for elem in df['Texte']:
        long.append(len(elem.split(' ')))
    df['longueur_texte'] = long
    # Filtre le Df pour ne conserver que les textes aynt un nb de mots sup à un seuil
    seuil = 15
    df = df[df['longueur_texte'] > seuil]

    # Création d'un corpus positif (Score > 3) et negatif (Score < 3)
    negatif = list(df[df['Score'] < 3]['Texte'])

    positif = list(df[df['Score'] > 3]['Texte'])
    return negatif, positif


"""
3) fonction Micro nettoyage
(input : str // Output: str)
"""


def micro_nett(texte):
    # Importer une liste de stopwords et normalisez la (accents, minuscule). Nous traitons des commentaires

    with open('stop-w_fr.txt') as f:
        sw = unidecode(f.read().lower()).split('\n')
    sw = sw + ['ci', "re", 'an', 'pr', 'it', 'ment', 'or', 'eme', 'su', 'dr', 'ere', 'mm', 'and', 'tt', "op", 'nait',
               'met']
    sw.remove('bon')
    sw.remove('bien')
    sw.remove('pas')
    sw.remove('trop')

    # normaliser le texte (minuscule et accents)
    texte = unidecode(texte).lower()
    texte_f = ''
    # suppression des caracteres spéciaux (:!/.;+-56336...)
    p = '[a-z]{1,}'
    for word in re.findall(p, unidecode(texte).lower()):
        # test si chaque mot restant n'est pas un stopword
        if word in sw:
            continue
        else:
            texte_f = texte_f + ' ' + word
    # retourne le texte nettoyé (chaque mot separé d'un expace)
    return texte_f

# La fonction 4 nexiste pas il s'agit d'une transformation

"""
5) Featurisation des corpus (WC,TFIDF)

input: Corpus nettoyé
Otput: Matrice BOW
"""

def feat(corpus):
    vectorizer = TfidfVectorizer(ngram_range=(2,4))
    X = vectorizer.fit_transform(corpus)
    Mat=pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())
    return Mat


"""
6) Extraction des combinaisons de mots les plus ii
input: Matrice BOW
output: str avec les combis mots avec meilleurs scores
"""


def extract(Mat):
    # Extraction par commentaire des meilleurs combiniaisons de mots (2/3 grams) selon leurs occurences
    # nombre de combinaisons par commentaire que je souhaite extraire
    nb = 3
    s = 0.2
    termes_pr = ''
    # Je balaie tous les commentaires
    for i in Mat.index:
        count = nb
        tmp = Mat.loc[i]
        # Tant que je n'ai pas trois resultats par commentaires je continue à collecter les combis
        while count != 0 and tmp[tmp.idxmax()] > s:
            # Je prends la colonne avec le meiller socre
            m = tmp.idxmax()
            # J'integre cette combinaison de mot dans ma chaine finale de resultats (touts les comments)
            # Je regroupe chaque mot de la colonne (n_gramm <=> n_mots) par un "_"
            termes_pr = termes_pr + ' ' + ' '.join([m.replace(' ', '_').strip()])
            # J'efface le max du commentaire pour pouvoir rechercher le second max
            del tmp[m]
            # Je decremente la valeur de count puisque je viens d'ajouter une combinaison de mots
            count = count - 1

    return termes_pr


"""

7) Creation des wordclouds et sauvegarde dans static/images

input: str avec les combis mots avec meilleurs scores
sauver les wordcloud dans static/images
output: O
"""


def wordcl(chainepos, chaineneg):
    wordcloud = WordCloud(prefer_horizontal=0.9,
                          width=400,
                          height=400, collocations=False
                          ).generate(chaineneg)
    wordcloud.to_file('static/images/N.png')

    wordcloud = WordCloud(prefer_horizontal=0.9,
                          width=400,
                          height=400, collocations=False
                          ).generate(chainepos)
    wordcloud.to_file('static/images/P.png')
