from flask import Flask,render_template,request
from tools import *

app = Flask(__name__)

@app.route('/')
def use():
    return render_template('home.html')

@app.route('/formulaire',methods=['POST'])
def pst():
    path_file='Scrap_Amazone.csv'
    if request.method == 'POST':
        # Je charge mon df
        df = load_csv(path_file)
        # Je manipule et filtre mon df
        neg, pos = macro_net(df)
        # Je nettoie chacun des textes de mes corpus
        positif = [micro_nett(elem) for elem in pos]
        negatif = [micro_nett(elem) for elem in neg]
        # Je créé mes matrices BOW (N-grams)N(Negatif) P(Positif)
        N = feat(negatif)
        P = feat(positif)
        # Je créé les chaines de caracteres à passer au Wordcloud
        pos_str = extract(P)
        neg_str = extract(N)
        # Je sauvegarde les wordcloud
        wordcl(pos_str, neg_str)

    return render_template('affiche.html')
    #return "{} {}".format(firstname,lastname)




if __name__ == "__main__":
    app.run(port="5000",debug=True)
