
from flask import Flask, render_template_string, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import random, os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sports.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Match(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    team_a = db.Column(db.String(50))
    team_b = db.Column(db.String(50))
    status = db.Column(db.String(20), default='Completed')

class PlayerStat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    match_id = db.Column(db.Integer, db.ForeignKey('match.id'))
    track_id = db.Column(db.Integer)
    team = db.Column(db.String(10))
    distance = db.Column(db.Float)
    max_speed = db.Column(db.Float)
    activity = db.Column(db.String(20))

with app.app_context():
    db.create_all()

@app.route('/')
def home():
    matches = Match.query.all()
    html = """
    <html><head><title>SportsAnalytics Pro</title>
    <style>body{font-family:sans-serif;background:#0a0a0a;color:white;padding:20px;} 
    .card{background:#1a1a1a;padding:20px;margin-bottom:10px;border-radius:8px;border:1px solid #333;}
    a.btn{background:#2563eb;color:white;padding:10px 15px;text-decoration:none;border-radius:5px;}</style></head>
    <body><h1>SportsAnalytics Pro</h1><h2>Match History</h2>
    {% for m in matches %}
    <div class="card"><h3>{{ m.team_a }} vs {{ m.team_b }}</h3>
    <p>Date: {{ m.date.strftime('%Y-%m-%d') }} | Status: {{ m.status }}</p>
    <a class="btn" href="/results/{{ m.id }}">View Results</a></div>
    {% else %}<p>No matches yet. <a href="/upload" style="color:#2563eb;">Upload one!</a></p>{% endfor %}
    </body></html>
    """
    return render_template_string(html, matches=matches)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        new_match = Match(team_a="NPFL Team A", team_b="NPFL Team B")
        db.session.add(new_match)
        db.session.commit()
        for i in range(1, 12):
            stat = PlayerStat(match_id=new_match.id, track_id=i, team="Team A", distance=random.uniform(8000, 11000), max_speed=random.uniform(25, 34), activity="JOGGING")
            db.session.add(stat)
        db.session.commit()
        return redirect(url_for('home'))
    return '''<html><body style="background:#0a0a0a;color:white;padding:20px;font-family:sans-serif;"><h1>Upload Match Video</h1><form method="POST"><input type="file" name="video"><br><br><button type="submit" style="padding:10px;background:#2563eb;color:white;border:none;border-radius:5px;">Process Video</button></form></body></html>'''

@app.route('/results/<int:match_id>')
def results(match_id):
    stats = PlayerStat.query.filter_by(match_id=match_id).all()
    html = """
    <html><head><title>Results</title>
    <style>body{font-family:sans-serif;background:#0a0a0a;color:white;padding:20px;}
    table{width:100%;border-collapse:collapse;margin-top:20px;}
    th,td{padding:10px;text-align:left;border-bottom:1px solid #333;}
    th{color:#888;}</style></head>
    <body><h1>Match Results</h1><a href="/" style="color:#2563eb;">&larr; Back to Matches</a>
    <table><tr><th>Track ID</th><th>Team</th><th>Distance (m)</th><th>Max Speed (km/h)</th><th>Activity</th></tr>
    {% for s in stats %}<tr><td>#{{ s.track_id }}</td><td>{{ s.team }}</td><td>{{ "%.0f"|format(s.distance) }}</td><td>{{ "%.1f"|format(s.max_speed) }}</td><td>{{ s.activity }}</td></tr>{% endfor %}
    </table></body></html>
    """
    return render_template_string(html, stats=stats)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
