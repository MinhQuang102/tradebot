# Coincex Telegram Bot

A Telegram bot for analyzing BTC/USD market data using technical indicators and providing trading signals.

## Setup

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables:
   - `TELEGRAM_TOKENS`: Comma-separated Telegram bot tokens
   - `NEWS_API_KEY`: NewsAPI key
   - `PORT`: 8080
4. Run the bot: `python QTVNEWVIPV2.py`

## Deployment on Render

1. Create a Web Service on [Render](https://render.com).
2. Link your GitHub repository.
3. Set environment variables in the Render dashboard:
   - `TELEGRAM_TOKENS`: Your bot token(s), comma-separated
   - `NEWS_API_KEY`: Your NewsAPI key
   - `PORT`: 8080
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python QTVNEWVIPV2.py`
   - **Instance Type**: Paid tier for 24/7 uptime
5. Deploy and monitor logs.

## Commands

- `/give`: Get market analysis after 5 seconds
- `/cskh`: Contact support (@mekiemtienlamnha)
- `/help`: Show available commands