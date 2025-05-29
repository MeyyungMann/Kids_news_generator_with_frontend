from newsapi import NewsApiClient
from config import Config

def test_newsapi():
    try:
        # Initialize the client
        newsapi = NewsApiClient(api_key=Config.NEWS_API_KEY)
        
        # Try to fetch some articles
        response = newsapi.get_top_headlines(
            category='business',
            language='en',
            page_size=1
        )
        
        print("NewsAPI Response Status:", response['status'])
        if response['status'] == 'ok':
            print("Successfully connected to NewsAPI!")
            if response.get('articles'):
                print("\nFirst article title:", response['articles'][0]['title'])
            else:
                print("No articles found")
        else:
            print("Error:", response.get('message', 'Unknown error'))
            
    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    test_newsapi()