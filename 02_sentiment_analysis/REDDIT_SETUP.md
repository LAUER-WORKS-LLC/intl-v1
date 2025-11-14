# Reddit API Setup Guide

Reddit now requires API authentication to access their data. Follow these steps to set up Reddit integration:

## Step 1: Create a Reddit Application

1. Go to https://www.reddit.com/prefs/apps
2. Click "create another app..." at the bottom
3. Fill in:
   - **Name**: Choose any name (e.g., "INT-L Sentiment Analysis")
   - **Type**: Select "script" (not web app)
   - **Description**: Optional
   - **About URL**: Leave blank
   - **Redirect URI**: Enter `http://localhost:8080` (required even for script apps)
4. Click "create app"

## Step 2: Get Your Credentials

After creating the app, you'll see a box with your app details. In that box:
- **Client ID**: The string **under your app name** (looks like: `abc123xyz456`) - NOT your username!
- **Client Secret**: The **"secret"** field (looks like: `def789uvw012`) - this is shown when you create the app, write it down!

**Important**: These are NOT your Reddit username and password. These are API credentials that Reddit generates for your app.

## Step 3: Add to .env File

Create or edit `.env` file in the project root:

```bash
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=INT-L-Sentiment-Analysis/1.0 (by /u/your_reddit_username)
```

### What Each Field Is:

1. **REDDIT_CLIENT_ID**: Copy from the box under your app name on Reddit (NOT your username!)
   - Example: `abc123xyz456789`

2. **REDDIT_CLIENT_SECRET**: Copy the "secret" field from Reddit app page
   - Example: `def789_uvw012xyz345`
   - **Important**: Write this down when you see it - you can only see it once when creating the app!

3. **REDDIT_USER_AGENT**: This is a **string you create yourself** - it's not located anywhere!
   - Format: `YourAppName/Version (by /u/your_reddit_username)`
   - Example: `INT-L-Sentiment-Analysis/1.0 (by /u/jlaue)`
   - Replace `your_reddit_username` with your actual Reddit username
   - This helps Reddit identify your application

## Step 4: Verify Setup

Run the sentiment analysis again. You should see:
```
✓ Reddit authenticated
```
or
```
✓ Reddit read access working
```

## Troubleshooting

### 401 Unauthorized Errors
- Double-check your Client ID and Secret are correct
- Make sure there are no extra spaces in the .env file
- Verify the app type is "script" not "web app"

### Rate Limits
- Reddit allows 60 requests per minute for authenticated apps
- The code includes automatic rate limiting (1 second delay between tickers)

### Still Having Issues?
- Try testing with a simple PRAW script first
- Check Reddit's API status: https://www.reddit.com/api/me.json
- Ensure your Reddit account is in good standing (not suspended)

