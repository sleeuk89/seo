import streamlit as st
import requests
from bs4 import BeautifulSoup
import spacy
from gensim import corpora, models
from collections import Counter
import matplotlib.pyplot as plt

# Initialize spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Set page config
st.set_page_config(page_title="SEO Research Tool", layout="wide")

def get_serp_results(keyword):
    """Get top 10 Google results using SerpStack API"""
    try:
        params = {
            "access_key": st.secrets["SERPSTACK_API_KEY"],
            "query": keyword,
            "num": 10
        }
        response = requests.get("http://api.serpstack.com/search", params=params)
        return [result["url"] for result in response.json()["organic_results"]]
    except Exception as e:
        st.error(f"Error fetching SERP results: {str(e)}")
        return []

def extract_page_data(url):
    """Scrape page content and structure"""
    try:
        response = requests.get(url, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        headings = {
            "h1": [h.text.strip() for h in soup.find_all('h1')],
            "h2": [h.text.strip() for h in soup.find_all('h2')],
            "h3": [h.text.strip() for h in soup.find_all('h3')]
        }
        
        paragraphs = [p.text.strip() for p in soup.find_all('p')]
        content = " ".join(paragraphs)
        
        domain = url.split('/')[2]
        internal_links = [
            a['href'] for a in soup.find_all('a', href=True) 
            if domain in a['href']
        ]
        
        return {
            "headings": headings,
            "content": content,
            "internal_links": internal_links,
            "url": url
        }
    except Exception as e:
        st.warning(f"Couldn't scrape {url}: {str(e)}")
        return None

@st.cache_data
def analyze_keywords(top_contents):
    """Extract NLP/LSI keywords from top content"""
    all_content = " ".join([c["content"] for c in top_contents if c])
    
    doc = nlp(all_content)
    nouns = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    verbs = [token.text.lower() for token in doc if token.pos_ == "VERB"]
    
    tfidf = Counter(nouns + verbs)
    
    texts = [[word for word in doc.text.lower().split()] for doc in nlp.pipe(all_content.split(". "))]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=5)
    
    return {
        "top_tfidf": tfidf.most_common(20),
        "lsi_keywords": [word[0] for word in lsi.show_topic(0, topn=10)] if lsi else []
    }

def main():
    st.title("Advanced SEO Research Tool")
    
    # User inputs
    keyword = st.text_input("Enter target keyword:")
    user_content = st.text_area("Paste your content here:", height=300)
    
    if st.button("Analyze"):
        if not keyword or not user_content:
            st.warning("Please fill both fields!")
            return
            
        with st.spinner("Analyzing SERP results..."):
            # Get SERP results
            serp_urls = get_serp_results(keyword)
            
            # Analyze competitors
            competitors = []
            for url in serp_urls[:5]:  # Analyze first 5 results
                data = extract_page_data(url)
                if data:
                    competitors.append(data)
            
            if not competitors:
                st.error("No competitors data found!")
                return
                
            # Keyword analysis
            keyword_data = analyze_keywords(competitors)
            
            # Visualization
            fig, ax = plt.subplots()
            ax.bar([k[0] for k in keyword_data["top_tfidf"]], [k[1] for k in keyword_data["top_tfidf"]])
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Content analysis
            user_keywords = [token.text.lower() for token in nlp(user_content) if token.pos_ in ["NOUN", "PROPN"]]
            competitor_keywords = [kw[0] for kw in keyword_data["top_tfidf"]]
            matched_keywords = set(user_keywords) & set(competitor_keywords)
            score = (len(matched_keywords) / len(set(competitor_keywords)) * 100 if len(set(competitor_keywords)) > 0 else 0
            
            # Show results
            st.subheader(f"SEO Score: {score:.1f}/100")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Top Recommended Keywords")
                st.write([k[0] for k in keyword_data["top_tfidf"][:10]])
                
                st.markdown("### Missing Keywords")
                missing_keywords = list(set(competitor_keywords) - set(user_keywords))[:10]
                st.write(missing_keywords if missing_keywords else ["All keywords matched!"])
                
            with col2:
                st.markdown("### Competitor Insights")
                for comp in competitors[:3]:
                    st.write(f"**URL**: {comp['url']}")
                    st.write(f"**H1 Headings**: {', '.join(comp['headings']['h1'])}")
                    st.write(f"**Content Length**: {len(comp['content']):,} characters")
                    st.write("---")

if __name__ == "__main__":
    main()
