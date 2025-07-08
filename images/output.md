```python
import sqlite3
import re

app = Flask(__name__)

app.secret_key = 'your-secret-key-here'

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            return render_template('login.html', error="Username and password required")

        conn = get_db_connection()
        user = conn.execute(
            'SELECT * FROM users WHERE username = ?',
            (username,)
        ).fetchone()

        if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash']):
            session['user_id'] = user['id']
            return redirect('/dashboard')
        else:
            return render_template('login.html', error="Invalid username and/or password")

app.secret_key = 'your-secret-key-here'
```