from flask import Flask, render_template, request
import logic
import interface

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(interface.chat(userText))
@app.route("/get_audio")
def get_bot_audio_response():
    return str(interface.Audio())

@app.route("/train")
def start():
    message = logic.train()
    return str(message)


if __name__ == "__main__":
    app.run()