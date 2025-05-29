import React, { useState } from 'react';
import axios from 'axios';

const WebSearch = ({ onGenerateArticle }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState('');

  const searchArticles = async () => {
    try {
      setIsSearching(true);
      setError('');
      const response = await axios.post('http://localhost:8000/search-articles', {
        query: searchQuery
      });
      setSearchResults(response.data.articles);
    } catch (err) {
      setError('Failed to search articles. Please try again.');
      console.error('Error:', err);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white shadow sm:rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h3 className="text-lg font-medium leading-6 text-gray-900">Search for Articles</h3>
          <div className="mt-2 max-w-xl text-sm text-gray-500">
            <p>Enter a topic or keywords to search for articles online.</p>
          </div>
          <div className="mt-5">
            <div className="flex space-x-2">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Enter search query..."
                className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
              />
              <button
                onClick={searchArticles}
                disabled={isSearching || !searchQuery}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50"
              >
                {isSearching ? 'Searching...' : 'Search'}
              </button>
            </div>
          </div>
        </div>
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

      {searchResults.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-medium text-gray-900">Search Results</h3>
          {searchResults.map((article, index) => (
            <div key={index} className="bg-white shadow overflow-hidden sm:rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <h4 className="text-lg font-medium text-gray-900">{article.title}</h4>
                <div className="mt-1 flex items-center space-x-2">
                  <p className="text-sm text-gray-500">{article.source}</p>
                  {(article.url || article.source) && (
                    <a
                      href={`https://${article.source}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-blue-600 hover:text-blue-800 hover:underline"
                    >
                      Read Original Article
                    </a>
                  )}
                </div>
                <div className="mt-2 text-sm text-gray-600">
                  {article.content.substring(0, 200)}...
                </div>
                <div className="mt-4 flex space-x-4">
                  <button
                    onClick={() => onGenerateArticle(article)}
                    className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-primary-700 bg-primary-100 hover:bg-primary-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
                  >
                    Generate Kid-Friendly Version
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default WebSearch; 