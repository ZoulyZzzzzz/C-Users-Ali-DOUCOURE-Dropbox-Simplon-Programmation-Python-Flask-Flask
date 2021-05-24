from flask import Flask,render_template,request
from werkzeug.utils import secure_filename
from tools import *
import config

app = Flask(__name__)

@app.route('/')
def use():
    return render_template(config.page_home)


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      path_file='data/'+secure_filename(f.filename)
      f.save(path_file)
      # Je charge mon df
      try:
        df = load_csv(path_file)
      except:
        return render_template('error_csv.html')


      # Je manipule et filtre mon df
      try:
          neg, pos = macro_net(df)
      except KeyError:
          return render_template('error_csv.html')
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






if __name__ == "__main__":
    app.run(port="5000",debug=True)
