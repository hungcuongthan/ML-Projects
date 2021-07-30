from flask import Flask, request, jsonify
import pickle as pkl

app = Flask(__name__)

clf = pkl.load(open('svc_model.pkl','rb'))


@app.route('/')
def hello_world():
	return 'hello_world'


@app.route('/get_sentiment', methods= ['POST'])
def get_sentiment():
	x = request.get_json(force = True)
	text = x['Review']

	sent = clf.predict([text])

	sent = sent[0]

	return jsonify(result = str(sent))

if __name__ == '__main__':
	app.run(
		# use_reloaded = True
		)


