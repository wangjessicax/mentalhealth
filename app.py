from flask import Flask


app = Flask(__name__)
@app.route('/questions')
def mentalhealth():





	return "hello"

if __name__ == '__main__':
		app.run(debug=True, use_reloader=True)