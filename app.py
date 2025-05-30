import os, json, time
import pandas as pd
import yfinance as yf
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import spacy
import pandas_ta as ta
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import json
from flask_caching import Cache
from datetime import datetime, date
from flask import Flask, render_template, request, redirect, url_for, flash
from datetime import datetime, timedelta
from models import predict_stock, sentiment_analysis
from database import init_db, get_db
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from textblob import TextBlob
from ticker_mapping import ORG_TO_TICKER
from dotenv import load_dotenv
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['DATABASE'] = os.path.join(app.instance_path, 'app.db')

# Load the NLP model for entity recognition
nlp = spacy.load('en_core_web_sm')

# Initialize the Sentiment Intensity Analyzer from NLTK
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Ensure instance folder exists
os.makedirs(app.instance_path, exist_ok=True)

# Initialize DB inside app context
with app.app_context():
    init_db()

# Configure caching
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 300})  # 5 minutes = 300 sec

# Setup login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = ''  # Suppress default login message

# User model for Flask-Login
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    if user:
        return User(user['id'], user['username'])
    return None

# Home route
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('main_dashboard'))
    return redirect(url_for('login'))

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()

        if user and check_password_hash(user['password'], password):
            user_obj = User(user['id'], user['username'])
            login_user(user_obj)
            return redirect(url_for('main_dashboard'))
        else:
            flash('Invalid username or password.', 'error')
    return render_template('auth/login.html')

# Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()

        existing_user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        if existing_user:
            flash('Username already exists. Choose a different one.', 'error')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        db.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
        db.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('auth/register.html')

# Main dashboard route
@app.route('/main_dashboard')
@login_required
def main_dashboard():
    return render_template('main_dashboard.html')


# Closing Prediction Analysis route
# Function to fetch Yahoo Finance data
def fetch_yahoo_data(ticker, start_date, end_date):
    try:
        today = date.today()
        if datetime.strptime(end_date, "%Y-%m-%d").date() > today:
            end_date = today.strftime("%Y-%m-%d")

        # Fetch historical stock data
        df = yf.download(ticker, start=start_date, end=end_date)

        if df.empty:
            raise ValueError(f"No data found for {ticker} between {start_date} and {end_date}")
        
        return df
    except Exception as e:
        logging.error(f"Error in fetch_yahoo_data: {e}")
        raise
# Sentiment Score Functions
def sentiment_analysis(ticker):
    try:
        NEWS_API_KEY = os.getenv('NEWSAPI_KEY')

        if not NEWS_API_KEY:
            raise ValueError("NewsAPI key not found in environment variables.")

        # Fetch news articles from NewsAPI
        url = (
            f'https://newsapi.org/v2/everything?'
            f'q={ticker}&'
            'sortBy=publishedAt&'
            'language=en&'
            f'apiKey={NEWS_API_KEY}'
        )
        
        response = requests.get(url)
        news_data = response.json()

        if news_data.get('status') != 'ok':
            raise ValueError(f"NewsAPI error: {news_data.get('message')}")

        articles = news_data.get('articles', [])

        if not articles:
            return {"score": 50, "verdict": "🤔 Neutral"}

        sentiment_total = 0
        count = 0

        for article in articles[:10]:  # Up to 10 latest articles
            title = article.get('title', '')
            description = article.get('description', '')
            combined_text = title + " " + description

            analysis = TextBlob(combined_text)
            sentiment_total += analysis.sentiment.polarity
            count += 1

        if count == 0:
            return {"score": 50, "verdict": "🤔 Neutral"}

        average_sentiment = sentiment_total / count
        normalized_sentiment = (average_sentiment + 1) * 50  # Normalize to 0-100

        score = round(normalized_sentiment, 2)

        # Add Dynamic Verdict
        if score > 60:
            verdict = "🚀 Positive"
        elif score < 30:
            verdict = "❌ Negative"
        else:
            verdict = "🤔 Neutral"

        return {"score": score, "verdict": verdict}

    except Exception as e:
        logging.error(f"Error in sentiment_analysis: {e}")
        return {"score": 50, "verdict": "🤔 Neutral"}

# Closing Prediction Analysis route
@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if request.method == 'POST':
        ticker = request.form['ticker']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        try:
            # Fetch stock data using Yahoo Finance API
            stock_data = fetch_yahoo_data(ticker, start_date, end_date)

            # If stock data is empty, show an error message
            if stock_data.empty:
                flash('No data found for the given ticker and date range.', 'error')
                return redirect(url_for('dashboard'))

            # Predict stock using multiple models
            prediction = predict_stock(stock_data, ticker)
            
            # Updated to get full sentiment dict (score + verdict)
            sentiment_result = sentiment_analysis(ticker)

            # Render results page
            return render_template(
                'results.html',
                prediction=prediction,
                sentiment=sentiment_result,  # <--- important: pass full dict
                ticker=ticker,
                start=start_date,
                end=end_date,
                stock_data=stock_data.to_json()
            )
        except Exception as e:
            flash(f"Error during analysis: {str(e)}", 'error')
            return redirect(url_for('dashboard'))

    # Render the dashboard template
    return render_template('dashboard.html')



# Buy/Sell Recommendation route
@app.route('/buy_sell', methods=['GET', 'POST'])
@login_required
def buy_sell():
    API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

    def get_sma_recommendation(ticker, time_period=10, tolerance_percent=2.0):
        try:
            sma_url = (
                f'https://www.alphavantage.co/query?function=SMA&symbol={ticker}'
                f'&interval=daily&time_period={time_period}&series_type=close&apikey={API_KEY}'
            )
            sma_data = requests.get(sma_url).json()
            recent_sma = float(list(sma_data['Technical Analysis: SMA'].values())[0]['SMA'])

            price_url = (
                f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={API_KEY}'
            )
            current_price = float(requests.get(price_url).json()['Global Quote']['05. price'])

            # Calculate tolerance range
            tolerance = (tolerance_percent / 100) * recent_sma
            upper_bound = recent_sma + tolerance
            lower_bound = recent_sma - tolerance

            if current_price > upper_bound:
                signal = "📈 Buy"
            elif current_price < lower_bound:
                signal = "📉 Sell"
            else:
                signal = "🤔 Hold"

            return f"SMA ({recent_sma:.2f}) vs Price ({current_price:.2f}): {signal}"
        except Exception as e:
            return f"SMA Error: {str(e)}"

    def get_rsi(ticker):
        try:
            url = f'https://www.alphavantage.co/query?function=RSI&symbol={ticker}&interval=daily&time_period=14&series_type=close&apikey={API_KEY}'
            data = requests.get(url).json()
            rsi = float(list(data['Technical Analysis: RSI'].values())[0]['RSI'])
            return rsi
        except Exception:
            return None

    def get_combined_recommendation(ticker):
        sma_reco = get_sma_recommendation(ticker)
        rsi = get_rsi(ticker)

        summary = [sma_reco]

        if rsi is not None:
            if rsi < 30:
                summary.append(f"RSI ({rsi:.2f}): 📈 Buy (oversold)")
            elif rsi > 60:
                summary.append(f"RSI ({rsi:.2f}): 📉 Sell (overbought)")
            else:
                summary.append(f"RSI ({rsi:.2f}): 🤔 Neutral")
        else:
            summary.append("RSI: Not Available")

        return " | ".join(summary)

    def generate_stock_chart(ticker):
        stock_data = yf.download(ticker, period="1y", interval="1d")
        plt.figure(figsize=(8, 4))
        plt.plot(stock_data['Close'], label='Closing Price')
        plt.title(f'{ticker} Stock Price')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend(loc='upper left')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def fetch_news(ticker):
        NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
        url = f'https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&apiKey={NEWSAPI_KEY}&language=en&pageSize=5'
        response = requests.get(url)
        news_items = []
        try:
            for article in response.json().get('articles', []):
                text = article.get('title', '') + ' ' + article.get('description', '')
                sentiment_score = TextBlob(text).sentiment.polarity
                sentiment = (
                    'positive' if sentiment_score > 0.2 else
                    'negative' if sentiment_score < -0.2 else
                    'neutral'
                )
                published_time = datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
                formatted_time = published_time.strftime('%b %d, %Y %I:%M %p')

                news_items.append({
                    'title': article['title'],
                    'description': article['description'],
                    'url': article['url'],
                    'publishedAt': formatted_time,
                    'sentiment': sentiment
                })
        except Exception as e:
            print("News fetch error:", str(e))
        return news_items

    if request.method == 'POST':
        ticker = request.form['ticker']

        recommendation = get_combined_recommendation(ticker)
        analysis = f"Based on SMA and RSI analysis for {ticker}: {recommendation}"

        chart_url = generate_stock_chart(ticker)
        news = fetch_news(ticker)

        current_time = datetime.now().strftime('%b %d, %Y %I:%M %p')

        return render_template('buy_sell_result.html',
                               recommendation=recommendation,
                               analysis=analysis,
                               ticker=ticker,
                               chart_url=chart_url,
                               news=news,
                               current_time=current_time)

    return render_template('buy_sell.html')




# Fetching Buzzing Stocks Data 
def fetch_buzzing_stocks():
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
    url = f'https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey={NEWSAPI_KEY}'
    response = requests.get(url)
    articles = response.json().get('articles', [])

    print("Fetched Articles:", len(articles))  # Debug log

    company_mentions = {}
    company_sentiments = {}

    for article in articles:
        title = article.get('title') or ''
        description = article.get('description') or ''
        text = title + ' ' + description

        sentiment = sia.polarity_scores(text)['compound']
        doc = nlp(text)

        print("Extracted ORGs:", [ent.text for ent in doc.ents if ent.label_ == 'ORG'])  # Debug log

        for ent in doc.ents:
            if ent.label_ == 'ORG':
                company = ent.text.strip()
                company_mentions[company] = company_mentions.get(company, 0) + 1
                sentiments = company_sentiments.get(company, [])
                sentiments.append(sentiment)
                company_sentiments[company] = sentiments

    buzzing_stocks = []

    for company, sentiments in company_sentiments.items():
        average_sentiment = sum(sentiments) / len(sentiments)
        mention_count = company_mentions.get(company, 0)

        # Relaxed filtering for testing — adjust back as needed
        if mention_count >= 1:
            if company in ORG_TO_TICKER:  # Check if company is in the dictionary
                ticker_symbol = ORG_TO_TICKER[company]
                try:
                    ticker = yf.Ticker(ticker_symbol)
                    stock_info = ticker.info
                    buzzing_stocks.append({
                        'company': company,
                        'mentions': mention_count,
                        'average_sentiment': average_sentiment,
                        'symbol': stock_info.get('symbol', 'N/A'),
                        'current_price': stock_info.get('regularMarketPrice', 'N/A')
                    })
                except Exception as e:
                    print(f"Error fetching stock data for {company}: {str(e)}")
                    continue
            else:
                print(f"Skipping invalid ticker for {company}: No matching stock symbol found.")

    buzzing_stocks.sort(key=lambda x: x['mentions'], reverse=True)
    print("Buzzing Stocks:", buzzing_stocks)  # Debug log
    return buzzing_stocks

# Route to display stocks in buzz
@app.route('/buzz_stocks')
def buzz_stocks():
    buzzing_stocks = fetch_buzzing_stocks()
    return render_template('buzz_stocks.html', stocks=buzzing_stocks)



# -------------------------------------
# Function to fetch IPO details from FINNHUB API
# -------------------------------------
def get_ipo_data():
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")  # Fetch API key from environment

    if not FINNHUB_API_KEY:
        print("FINNHUB_API_KEY not found.")
        return []

    url = f"https://finnhub.io/api/v1/calendar/ipo?from=2025-01-01&to=2025-12-31&token={FINNHUB_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        if isinstance(data, dict) and "ipoCalendar" in data:
            return data["ipoCalendar"]  # 👈 Returns list directly
        else:
            print("Unexpected response format:", data)
            return []
    except Exception as e:
        print("Error fetching IPO data:", e)
        return []

# -------------------------------------
# Function to filter IPOs for the previous 2 days and the upcoming 7 days
# -------------------------------------
def get_ipo_for_window(ipo_data):
    today = datetime.today()
    two_days_ago = today - timedelta(days=2)
    one_week_later = today + timedelta(days=7)

    ipo_in_window = []

    for ipo in ipo_data:
        try:
            ipo_date = datetime.strptime(ipo['date'], '%Y-%m-%d')
            if two_days_ago <= ipo_date <= one_week_later:
                ipo_in_window.append(ipo)
        except Exception as e:
            print(f"Date error for IPO: {ipo['date']}, {e}")

    return ipo_in_window

# -------------------------------------
# Verdict Logic for IPOs (BUY / AVOID / NEUTRAL)
# -------------------------------------
def assign_verdict(ipo):
    # Extract relevant data
    price_range = ipo.get('price', '')
    shares_offered = ipo.get('numberOfShares', 0)
    exchange = ipo.get('exchange', 'N/A')

    # Handle price range parsing
    if isinstance(price_range, str):
        try:
            if '-' in price_range:
                min_price, max_price = map(float, price_range.split('-'))
            else:
                min_price = max_price = float(price_range)
        except ValueError:
            min_price = max_price = 0.0
    else:
        min_price = max_price = float(price_range) if price_range else 0.0

    # Initialize scores
    price_score = 0
    shares_score = 0
    exchange_score = 0

    # Price-based scoring
    if max_price < 10:
        price_score = 1  # Good
    elif min_price > 50:
        price_score = -1  # Risky
    else:
        price_score = 0  # Neutral

    # Shares-based scoring
    if shares_offered < 5_000_000:
        shares_score = 1  # Good
    elif shares_offered > 20_000_000:
        shares_score = -1  # Risky
    else:
        shares_score = 0  # Neutral

    # Exchange-based scoring
    trusted_exchanges = ['nasdaq', 'nasdaq global', 'nasdaq capital', 'nyse']
    if exchange.lower() in trusted_exchanges:
        exchange_score = 0  # Neutral (safe)
    else:
        exchange_score = -1  # Risky

    # Now final verdict
    total_score = price_score + shares_score + exchange_score

    if price_score == 1 and shares_score == 1 and exchange_score == 0:
        final_verdict = "🚀 STRONG BUY"
    elif price_score == -1 or shares_score == -1 or exchange_score == -1:
        final_verdict = "❌ AVOID"
    elif total_score > 0:
        final_verdict = "✅ BUY"
    else:
        final_verdict = "🤔 NEUTRAL"

    return final_verdict


# -------------------------------------
# IPO page route
# -------------------------------------
@app.route('/ipo')
def ipo_page():
    ipo_data = get_ipo_data()

    if ipo_data:
        ipo_data_in_window = get_ipo_for_window(ipo_data)

        for ipo in ipo_data_in_window:
            # Format the date
            try:
                ipo['date'] = datetime.strptime(ipo['date'], '%Y-%m-%d').strftime('%B %d, %Y')
            except Exception as e:
                print(f"Date formatting error: {e}")
                ipo['date'] = ipo.get('date', 'N/A')

            # Add Verdict
            ipo['verdict'] = assign_verdict(ipo)

        return render_template('ipo_page.html', ipo_data=ipo_data_in_window)
    else:
        return render_template('ipo_page.html', ipo_data=None)
    

# Screener Options 

# ====== Function to fetch Most Active US Stocks ======
@cache.cached(timeout=300)  # cache for 5 minutes
def get_most_active_stocks_us():
    url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
    params = {
        "formatted": "true",
        "scrIds": "most_actives",
        "count": 10
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }
    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()

        symbols = []
        quotes = data['finance']['result'][0]['quotes']
        for q in quotes:
            symbols.append({
                'symbol': q.get('symbol', 'N/A'),
                'shortName': q.get('shortName', 'N/A'),
                'regularMarketPrice': q['regularMarketPrice']['raw'] if q.get('regularMarketPrice') else None,
                'volume': q['regularMarketVolume']['raw'] if q.get('regularMarketVolume') else None
            })
        return symbols
    except requests.exceptions.RequestException as e:
        print(f"Error fetching most active US stocks: {e}")
        return []


# ====== Flask Route for High Volume Screener ======
@app.route('/screener/high-volume')
@login_required
def high_volume_stocks():
    us_stocks = get_most_active_stocks_us()
    return render_template('high_volume.html', us_stocks=us_stocks)


# ------- Function for 52 week High ------

# -------- Cache and Files --------
# -------- Constants --------
CACHE_FILE = '52_week_high_cache.json'
TICKER_FILE_US = 'tickers_us.txt'
TICKER_FILE_NSE = 'tickers_nse.txt'

# -------- Read tickers --------
def read_tickers(file):
    if not os.path.exists(file):
        raise FileNotFoundError(f"Ticker file '{file}' not found.")
    with open(file, 'r') as f:
        return [line.strip().upper() for line in f if line.strip()]

# -------- Check if cache is valid (1 hour) --------
def is_cache_valid(file):
    if not os.path.exists(file):
        return False
    last_modified = datetime.fromtimestamp(os.path.getmtime(file))
    age_seconds = (datetime.now() - last_modified).total_seconds()
    return age_seconds < 3600

# -------- Load cache data --------
def load_cache(file):
    with open(file, 'r') as f:
        return json.load(f)

# -------- Save cache data --------
def save_cache(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

# -------- Fetch stock data --------
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty:
            return None, None
        high = round(hist['High'].max(), 2)
        today = round(hist['Close'].iloc[-1], 2)
        if today > high:
            high = today
        return high, today
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None, None

# -------- Flask route for 52-week high stocks --------
@app.route('/screener/52-week-high')
def fifty_two_week_high_stocks():
    tickers_us = read_tickers(TICKER_FILE_US)
    tickers_nse = read_tickers(TICKER_FILE_NSE)

    # Load cache or build new one
    if is_cache_valid(CACHE_FILE):
        print("Using cached data.")
        try:
            cache = load_cache(CACHE_FILE)
        except Exception as e:
            print("Error loading cache. Rebuilding...", e)
            cache = {}
    else:
        print("Building new cache...")
        cache = {}

        for ticker in tickers_us:
            high, today = fetch_stock_data(ticker)
            if high and today:
                cache[ticker] = {
                    '52w_high': high,
                    'today': today,
                    'market': 'US'
                }

        for ticker in tickers_nse:
            high, today = fetch_stock_data(ticker)
            if high and today:
                cache[ticker] = {
                    '52w_high': high,
                    'today': today,
                    'market': 'NSE'
                }

        print("Saving cache...")
        save_cache(cache, CACHE_FILE)

    data = []
    for ticker, values in cache.items():
        if '52w_high' not in values or 'today' not in values:
            print(f"Skipping {ticker}: missing keys in cache.")
            continue

        high = values['52w_high']
        today = values['today']
        percent_diff = abs(today - high) / high * 100

        if percent_diff <= 10:
            try:
                info = yf.Ticker(ticker).info
                name = info.get('shortName', 'N/A')
            except Exception:
                name = 'N/A'

            data.append({
                'symbol': ticker,
                'name': name,
                'today': today,
                'high': high,
                'percent_diff': round(percent_diff, 2),
                'market': values['market']
            })

    # Sort and separate by market
    data.sort(key=lambda x: x['percent_diff'])

    data_us = [d for d in data if d['market'] == 'US']
    data_nse = [d for d in data if d['market'] == 'NSE']

    return render_template('52_week_high.html', data_us=data_us, data_nse=data_nse)



# ------- Function for 52 week Low ------
# ------- Constants -------
CACHE_FILE = '52_week_low_cache.json'
TICKER_FILE_US = 'tickers_us.txt'
TICKER_FILE_NSE = 'tickers_nse.txt'

# -------- Read tickers --------
def read_tickers(file):
    if not os.path.exists(file):
        return []
    with open(file, 'r') as f:
        return [line.strip().upper() for line in f if line.strip()]

# -------- Check if cache is valid (1 hour) --------
def is_cache_valid(file):
    if not os.path.exists(file):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(file))
    return age.total_seconds() < 3600

# -------- Load cache data --------
def load_cache(file):
    try:
        with open(file, 'r') as f:
            return json.load(f)
    except Exception:
        return {}  # Fallback to empty if corrupted

# -------- Save cache data --------
def save_cache(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

# -------- Fetch stock data --------
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty:
            return None, None
        low = round(hist['Low'].min(), 2)
        today = round(hist['Close'].iloc[-1], 2)
        return low, today
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None, None

# -------- Flask route for 52-week low stocks --------
@app.route('/screener/52-week-low')
def fifty_two_week_low_stocks():
    tickers_us = read_tickers(TICKER_FILE_US)
    tickers_nse = read_tickers(TICKER_FILE_NSE)

    if is_cache_valid(CACHE_FILE):
        cache = load_cache(CACHE_FILE)
    else:
        cache = {}
        for ticker in tickers_us:
            low, today = fetch_stock_data(ticker)
            if low is not None and today is not None:
                cache[ticker] = {'52w_low': low, 'today': today, 'market': 'US'}
        for ticker in tickers_nse:
            low, today = fetch_stock_data(ticker)
            if low is not None and today is not None:
                cache[ticker] = {'52w_low': low, 'today': today, 'market': 'NSE'}
        save_cache(cache, CACHE_FILE)

    data_us = []
    data_nse = []

    for ticker, values in cache.items():
        if '52w_low' not in values or 'today' not in values:
            continue  # skip incomplete entries

        low = values['52w_low']
        today = values['today']
        if low == 0:
            continue
        percent_diff = abs(today - low) / low * 100
        if percent_diff <= 5:
            try:
                name = yf.Ticker(ticker).info.get('shortName', 'N/A')
            except Exception:
                name = 'N/A'

            stock_data = {
                'symbol': ticker,
                'name': name,
                'today': today,
                'low': low,
                'percent_diff': round(percent_diff, 2)
            }

            if values.get('market') == 'US':
                data_us.append(stock_data)
            elif values.get('market') == 'NSE':
                data_nse.append(stock_data)

    data_us.sort(key=lambda x: x['percent_diff'])
    data_nse.sort(key=lambda x: x['percent_diff'])

    return render_template('52_week_low.html', data_us=data_us, data_nse=data_nse)

# -------- Dividend & Payout Ratio Screener --------
# -------- Cache and Files --------
DIVIDEND_CACHE_FILE = 'high_dividend_cache.json'
TICKER_FILE_US = 'tickers_us.txt'  
TICKER_FILE_NSE = 'tickers_nse.txt' 

# -------- Read tickers --------
def read_tickers(file):
    if not os.path.exists(file):
        print(f"Ticker file '{file}' not found.")
        return []
    with open(file, 'r') as f:
        return [line.strip().replace('$', '').upper() for line in f if line.strip()]

# -------- Cache handling --------
def is_cache_valid(file):
    if not os.path.exists(file):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(file))
    return age.total_seconds() < 3600  # 1 hour validity

def load_cache(file):
    with open(file, 'r') as f:
        return json.load(f)

def save_cache(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

# -------- Fetch dividend and payout ratio --------
def fetch_dividend_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        dividend = info.get("dividendRate")
        price = info.get("currentPrice")
        eps = info.get("trailingEps")

        print(f"Fetching {ticker}: Dividend={dividend}, Price={price}, EPS={eps}")

        if dividend and price:
            dividend_yield = round((dividend / price) * 100, 2)
            payout_ratio = round((dividend / eps) * 100, 2) if eps and eps != 0 else "N/A"

            return {
                "symbol": ticker,
                "name": info.get("shortName", "N/A"),
                "price": round(price, 2),
                "yield": dividend_yield,
                "payout_ratio": payout_ratio
            }
        else:
            return None
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

# -------- Flask route --------
@app.route('/screener/high-dividend')
def high_dividend_stocks():
    tickers_us = read_tickers(TICKER_FILE_US)
    tickers_nse = read_tickers(TICKER_FILE_NSE)

    if is_cache_valid(DIVIDEND_CACHE_FILE):
        cached_data = load_cache(DIVIDEND_CACHE_FILE)
        data_us = cached_data.get('us', [])
        data_nse = cached_data.get('nse', [])
    else:
        data_us = []
        for ticker in tickers_us:
            stock_data = fetch_dividend_data(ticker)
            if stock_data and stock_data["yield"] >= 4.0:
                data_us.append(stock_data)

        data_nse = []
        for ticker in tickers_nse:
            stock_data = fetch_dividend_data(ticker)
            if stock_data and stock_data["yield"] >= 4.0:
                data_nse.append(stock_data)

        save_cache({"us": data_us, "nse": data_nse}, DIVIDEND_CACHE_FILE)

    data_us.sort(key=lambda x: x["yield"], reverse=True)
    data_nse.sort(key=lambda x: x["yield"], reverse=True)

    return render_template("high_dividend.html", data_us=data_us, data_nse=data_nse)




#------ Fetching & Routing global etf's -----
ETF_SYMBOLS = [
    'SPY', 'IVV', 'VTI', 'QQQ', 'EFA', 'VWO', 'IWM', 'EEM', 'AGG', 'BND',
]

def fetch_global_etfs(tickers):
    data = []
    for symbol in tickers:
        try:
            info = yf.Ticker(symbol).info
            data.append({
                'symbol': symbol,
                'name': info.get('shortName', 'N/A'),
                'price': info.get('regularMarketPrice', 'N/A'),
                'volume': info.get('volume', 'N/A')
            })
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            data.append({
                'symbol': symbol,
                'name': 'N/A',
                'price': 'N/A',
                'volume': 'N/A'
            })
    return data

@app.route('/screener/global-etfs')
def global_etfs():
    etfs = fetch_global_etfs(ETF_SYMBOLS)
    return render_template('global_etfs.html', etfs=etfs)

# Screener Route
@app.route('/screener')
def screener():
    return render_template('screener.html')



# Logout route
@app.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
