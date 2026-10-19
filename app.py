
from flask import Flask, render_template, send_from_directory

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('demo.html')

@app.route('/event')
def event():
    return render_template('event.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/video_feed')
def video_feed():
    return send_from_directory('static', 'video.mp4')

@app.route('/health')
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(debug=True)
