import React, { useState, useEffect } from 'react';
import axios from 'axios';
import FeedbackForm from '../components/FeedbackForm';

export const Home = () => {
  const [topic, setTopic] = useState('');
  const [ageGroup, setAgeGroup] = useState('3-6');
  const [category, setCategory] = useState('Science');
  const [includeGlossary, setIncludeGlossary] = useState(true);
  const [generateImage, setGenerateImage] = useState(true);
  const [selectedLanguages, setSelectedLanguages] = useState(['en']);
  const [loading, setLoading] = useState(false);
  const [article, setArticle] = useState(null);
  const [error, setError] = useState('');
  const [articles, setArticles] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setArticle(null);

    try {
      const response = await axios.post('http://localhost:8000/generate', {
        topic,
        age_group: parseInt(ageGroup.split('-')[0]),
        category,
        include_glossary: includeGlossary,
        generate_image: generateImage,
        target_languages: selectedLanguages
      });
      setArticle(response.data);
    } catch (err) {
      setError('Failed to generate article. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchNews = async (category) => {
    setLoading(true);
    setError('');
    try {
      const response = await axios.get(`http://localhost:8000/news/${category.toLowerCase()}`);
      setArticles(response.data.articles);
    } catch (err) {
      setError('Failed to fetch news articles. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (category) {
      fetchNews(category);
    }
  }, [category]);

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-gray-900 mb-8">Kid-Friendly News</h1>
      
      <div className="mb-6">
        <label htmlFor="category" className="block text-sm font-medium text-gray-700">
          Category
        </label>
        <select
          id="category"
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
        >
          <option value="Science">Science</option>
          <option value="Technology">Technology</option>
          <option value="Health">Health</option>
          <option value="Environment">Environment</option>
          <option value="Economy">Economy</option>
        </select>
      </div>

      {error && (
        <div className="mt-4 rounded-md bg-red-50 p-4">
          <div className="flex">
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">{error}</h3>
            </div>
          </div>
        </div>
      )}

      {loading ? (
        <div className="text-center py-4">Loading articles...</div>
      ) : (
        <div className="space-y-6">
          {articles.map((article, index) => (
            <div key={index} className="bg-white rounded-lg shadow overflow-hidden">
              <div className="p-6">
                <h2 className="text-2xl font-bold text-gray-900 mb-2">{article.title}</h2>
                <p className="text-sm text-gray-500 mb-4">
                  Source: {article.source} • {new Date(article.published_at).toLocaleDateString()}
                </p>
                <div className="prose max-w-none">
                  {article.content.split('\n').map((paragraph, i) => (
                    <p key={i} className="mb-4">{paragraph}</p>
                  ))}
                </div>
                <div className="mt-6 border-t pt-6">
                  <FeedbackForm
                    articleId={article.id}
                    ageGroup={article.age_group || parseInt(ageGroup.split('-')[0])}
                    category={article.category}
                  />
                </div>
                <a
                  href={article.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-4 inline-block text-indigo-600 hover:text-indigo-500"
                >
                  Read original article →
                </a>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}; 