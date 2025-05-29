// npm run dev -- --host

import React, { useState } from 'react';
import axios from 'axios';
import ArticleHistory from './components/ArticleHistory';
import WebSearch from './components/WebSearch';
import './App.css';

function App() {
  const [loading, setLoading] = useState(false);
  const [article, setArticle] = useState(null);
  const [error, setError] = useState('');
  const [selectedAgeGroup, setSelectedAgeGroup] = useState('7-9'); // Default age group
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [realNews, setRealNews] = useState(null);
  const [activeTab, setActiveTab] = useState('generate');

  const categories = [
    'Economy',
    'Science',
    'Technology',
    'Health',
    'Environment'
  ];

  const ageGroups = [
    { value: '3-6', label: 'Ages 3-6 (Preschool)' },
    { value: '7-9', label: 'Ages 7-9 (Early Elementary)' },
    { value: '10-12', label: 'Ages 10-12 (Upper Elementary)' }
  ];

  const getAgeGroupNumber = (ageRange) => {
    const [min] = ageRange.split('-').map(Number);
    return min;
  };

  const fetchRealNews = async () => {
    if (!selectedCategory) {
      setError('Please select a category first');
      return;
    }

    setLoading(true);
    setError('');
    setRealNews(null);

    try {
      const response = await axios.get(`http://localhost:8000/news/${selectedCategory}`);
      if (response.data.articles && response.data.articles.length > 0) {
        const article = response.data.articles[0];
        
        // Only try to get full content if we have a valid article ID
        if (article.id) {
          try {
            const fullContentResponse = await axios.get(`http://localhost:8000/news/${selectedCategory}/full/${article.id}`);
            if (fullContentResponse.data && fullContentResponse.data.content) {
              article.content = fullContentResponse.data.content;
            }
          } catch (err) {
            console.warn('Could not fetch full content:', err);
          }
        }
        
        setRealNews(article);
      } else {
        setError('No news articles found for this category.');
      }
    } catch (err) {
      setError('Failed to fetch news. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const generateArticle = async () => {
    if (!selectedCategory) {
      setError('Please select a category first');
      return;
    }

    setLoading(true);
    setError('');
    setArticle(null);

    try {
      // First try to fetch real news
      await fetchRealNews();
      
      // Prepare the request payload
      const requestPayload = {
        topic: selectedCategory,
        age_group: getAgeGroupNumber(selectedAgeGroup),
        category: selectedCategory,
        include_glossary: true,
        generate_image: true
      };

      // If we have real news, add it to the payload
      if (realNews) {
        requestPayload.original_news = realNews;
      }

      // Generate kid-friendly content
      const response = await axios.post('http://localhost:8000/generate', requestPayload);
      setArticle(response.data);
    } catch (err) {
      setError('Failed to generate article. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleWebSearchGenerate = async (article) => {
    try {
      setLoading(true);
      setError('');
      console.log('Generating from article:', article);
      
      // Use source as URL if url is null and ensure it has https:// prefix
      const articleUrl = article.url || `https://${article.source}`;
      
      if (!articleUrl) {
        setError('No valid URL found for this article');
        return;
      }
      
      const response = await axios.post('http://localhost:8000/generate-from-url', {
        url: articleUrl,  // Use the properly formatted URL
        title: article.title,
        source: article.source,
        age_group: getAgeGroupNumber(selectedAgeGroup),
        generate_image: true
      });
      
      console.log('Generation response:', response.data);
      setArticle(response.data);
      setActiveTab('generate');
    } catch (err) {
      console.error('Generation error:', err);
      setError('Failed to generate article. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="bg-white shadow-sm">
        <div className="max-w-4xl mx-auto px-4">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <h1 className="text-xl font-bold text-gray-800">Kids News Generator</h1>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setActiveTab('generate')}
                className={`px-4 py-2 rounded-md text-sm font-medium ${
                  activeTab === 'generate'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Generate
              </button>
              <button
                onClick={() => setActiveTab('web-search')}
                className={`px-4 py-2 rounded-md text-sm font-medium ${
                  activeTab === 'web-search'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Web Search
              </button>
              <button
                onClick={() => setActiveTab('history')}
                className={`px-4 py-2 rounded-md text-sm font-medium ${
                  activeTab === 'history'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                History
              </button>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-4xl mx-auto py-6 px-4">
        {activeTab === 'generate' ? (
          <div>
            <div className="min-h-screen bg-gray-50">
              <header className="bg-white shadow">
                <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
                  <h1 className="text-3xl font-bold text-gray-900">Kids News Generator</h1>
                </div>
              </header>

              <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
                <div className="px-4 py-6 sm:px-0">
                  <div className="bg-white rounded-lg shadow p-6">
                    <div className="space-y-4">
                      {/* Category Selection */}
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-4">
                          Select a category for the news article
                        </label>
                        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                          {categories.map((category) => (
                            <button
                              key={category}
                              onClick={() => setSelectedCategory(category)}
                              className={`inline-flex items-center justify-center px-4 py-2 border text-sm font-medium rounded-md shadow-sm ${
                                selectedCategory === category
                                  ? 'bg-primary-600 text-white border-transparent'
                                  : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                              }`}
                            >
                              {category}
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* Age Group Selection */}
                      <div className="mt-6">
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Select Age Group
                        </label>
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                          {ageGroups.map((ageGroup) => (
                            <button
                              key={ageGroup.value}
                              onClick={() => setSelectedAgeGroup(ageGroup.value)}
                              className={`inline-flex items-center justify-center px-4 py-2 border text-sm font-medium rounded-md shadow-sm ${
                                selectedAgeGroup === ageGroup.value
                                  ? 'bg-primary-600 text-white border-transparent'
                                  : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                              }`}
                            >
                              {ageGroup.label}
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* Add Generate Button after age group selection */}
                      <div className="mt-6 flex justify-center">
                        <button
                          onClick={generateArticle}
                          disabled={loading || !selectedCategory}
                          className={`inline-flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 ${
                            (loading || !selectedCategory) ? 'opacity-50 cursor-not-allowed' : ''
                          }`}
                        >
                          {loading ? 'Generating...' : 'Generate Article'}
                        </button>
                      </div>

                      {error && (
                        <div className="rounded-md bg-red-50 p-4">
                          <div className="flex">
                            <div className="ml-3">
                              <h3 className="text-sm font-medium text-red-800">{error}</h3>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Add Real News Preview */}
                      {realNews && !article && (
                        <div className="mt-6 space-y-6">
                          <div className="bg-white shadow overflow-hidden sm:rounded-lg">
                            <div className="px-4 py-5 sm:px-6">
                              <h3 className="text-lg leading-6 font-medium text-gray-900">Original News</h3>
                              <p className="mt-1 max-w-2xl text-sm text-gray-500">
                                Source: {realNews.source} | Published: {new Date(realNews.published_at).toLocaleDateString()}
                              </p>
                            </div>
                            <div className="border-t border-gray-200">
                              <div className="px-4 py-5 sm:p-6">
                                <h4 className="text-lg font-medium text-gray-900">{realNews.title}</h4>
                                <div className="mt-2 text-gray-600 whitespace-pre-wrap">{realNews.content}</div>
                                {realNews.url && (
                                  <a 
                                    href={realNews.url} 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                    className="mt-4 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-primary-700 bg-primary-100 hover:bg-primary-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
                                  >
                                    Read Original Article
                                  </a>
                                )}
                              </div>
                            </div>
                          </div>
                        </div>
                      )}

                      {article && (
                        <div className="mt-6 space-y-6">
                          <div className="bg-white shadow overflow-hidden sm:rounded-lg">
                            <div className="px-4 py-5 sm:px-6">
                              <h3 className="text-lg leading-6 font-medium text-gray-900">{article.title}</h3>
                              <p className="mt-1 max-w-2xl text-sm text-gray-500">
                                Category: {article.category} | Reading Level: {article.reading_level}
                              </p>
                            </div>
                            <div className="border-t border-gray-200">
                              <div className="px-4 py-5 sm:p-6 prose max-w-none">
                                {article.image_url && (
                                  <img
                                    src={`http://localhost:8000${article.image_url}`}
                                    alt={article.title}
                                    className="w-full max-h-[500px] object-contain rounded-lg mb-6 shadow-md"
                                  />
                                )}
                                <div className="whitespace-pre-wrap">{article.content}</div>
                                
                                {/* Updated original article link section */}
                                {article.original_article && (
                                  <div className="mt-6 pt-4 border-t border-gray-200">
                                    <div className="flex flex-col space-y-4">
                                      <div>
                                        <h4 className="text-lg font-medium text-gray-900">Source Article</h4>
                                        <p className="text-sm text-gray-500">
                                          {article.original_article.source} • {new Date(article.original_article.published_at).toLocaleDateString()}
                                        </p>
                                      </div>
                                      <div className="flex flex-col space-y-2">
                                        <a
                                          href={article.original_article.url}
                                          target="_blank"
                                          rel="noopener noreferrer"
                                          className="inline-flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors duration-200"
                                        >
                                          Read Original Article
                                        </a>
                                        <p className="text-sm text-gray-500 text-center">
                                          Opens in a new tab
                                        </p>
                                      </div>
                                    </div>
                                  </div>
                                )}
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </main>
            </div>
          </div>
        ) : activeTab === 'web-search' ? (
          <WebSearch onGenerateArticle={handleWebSearchGenerate} />
        ) : (
          <ArticleHistory />
        )}
      </main>
    </div>
  );
}

export default App; 