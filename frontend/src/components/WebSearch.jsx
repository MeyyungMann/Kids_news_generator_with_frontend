import React, { useState } from 'react';
import axios from 'axios';
import FeedbackForm from './FeedbackForm';

const WebSearch = ({ onGenerateArticle }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState('');
  const [selectedAgeGroup, setSelectedAgeGroup] = useState('7-9'); // Default age group
  const [generatedArticle, setGeneratedArticle] = useState(null);

  const ageGroups = [
    { value: '3-6', label: 'Ages 3-6 (Preschool)' },
    { value: '7-9', label: 'Ages 7-9 (Early Elementary)' },
    { value: '10-12', label: 'Ages 10-12 (Upper Elementary)' }
  ];

  const searchArticles = async () => {
    try {
      setIsSearching(true);
      setError('');
      const response = await axios.post('http://localhost:8000/api/search-articles', {
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

  const handleGenerateArticle = async (article) => {
    try {
      const generated = await onGenerateArticle({ ...article, ageGroup: selectedAgeGroup });
      setGeneratedArticle(generated);
    } catch (err) {
      setError('Failed to generate article. Please try again.');
      console.error('Error:', err);
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
                    onClick={() => handleGenerateArticle(article)}
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

      {generatedArticle && (
        <div className="bg-white shadow overflow-hidden sm:rounded-lg">
          <div className="px-4 py-5 sm:p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Generated Kid-Friendly Article</h3>
            {generatedArticle.image_url && (
              <img
                src={`http://localhost:8000${generatedArticle.image_url}`}
                alt={generatedArticle.title}
                className="mt-4 rounded-lg shadow-md w-full max-w-2xl mx-auto"
                onError={(e) => {
                  e.target.onerror = null;
                  e.target.src = 'path/to/fallback/image.png';
                }}
              />
            )}
            <div className="prose max-w-none">
              {generatedArticle.content.split('\n').map((paragraph, i) => (
                <p key={i} className="mb-4">{paragraph}</p>
              ))}
            </div>
            <div className="mt-6 border-t pt-6">
              <FeedbackForm
                articleId={generatedArticle.id || generatedArticle.topic?.replace(/[^a-zA-Z0-9]/g, '_') + '_' + Date.now()}
                ageGroup={parseInt(selectedAgeGroup.split('-')[0])}
                category={generatedArticle.category || 'General'}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default WebSearch; 