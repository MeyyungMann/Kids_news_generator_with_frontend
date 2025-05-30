import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ArticleHistory = () => {
    const [articles, setArticles] = useState([]);
    const [selectedCategory, setSelectedCategory] = useState('all');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [showPrompts, setShowPrompts] = useState({});  // Track which articles show prompts

    const categories = [
        { id: 'all', name: 'All Categories' },
        { id: 'Science', name: 'Science' },
        { id: 'Technology', name: 'Technology' },
        { id: 'Health', name: 'Health' },
        { id: 'Environment', name: 'Environment' },
        { id: 'Economy', name: 'Economy' }
    ];

    useEffect(() => {
        fetchArticles();
    }, [selectedCategory]);

    const fetchArticles = async () => {
        try {
            setLoading(true);
            setError(null);
            const response = await axios.get(`/api/articles/history?category=${selectedCategory}`);
            const articlesData = response.data.articles || [];
            setArticles(articlesData);
            
            // Initialize showPrompts state for new articles
            const newShowPrompts = {};
            articlesData.forEach((_, index) => {
                newShowPrompts[index] = false;
            });
            setShowPrompts(newShowPrompts);
        } catch (err) {
            setError('Failed to fetch articles. Please try again later.');
            console.error('Error fetching articles:', err);
            setArticles([]);
            setShowPrompts({});
        } finally {
            setLoading(false);
        }
    };

    const togglePrompts = (index) => {
        setShowPrompts(prev => ({
            ...prev,
            [index]: !prev[index]
        }));
    };

    const cleanText = (text) => {
        if (!text) return "";
        
        // Remove any prompt-related content
        const lines = text.split('\n');
        const cleanedLines = lines.filter(line => {
            const lowerLine = line.toLowerCase();
            return !lowerLine.includes('guidelines:') &&
                   !lowerLine.includes('now, create') &&
                   !lowerLine.includes('example:') &&
                   !lowerLine.includes('you are') &&
                   !lowerLine.includes('style:') &&
                   !lowerLine.includes('negative_prompt:') &&
                   !lowerLine.includes('format the response');
        });
        
        // Join the lines and clean up extra whitespace
        return cleanedLines.join('\n')
            .replace(/\s+/g, ' ')
            .trim();
    };

    return (
        <div className="max-w-4xl mx-auto p-4">
            <div className="mb-6">
                <h2 className="text-2xl font-bold mb-4">Article History</h2>
                <div className="flex flex-wrap gap-2">
                    {categories.map(category => (
                        <button
                            key={category.id}
                            onClick={() => setSelectedCategory(category.id)}
                            className={`px-4 py-2 rounded-full text-sm font-medium transition-colors
                                ${selectedCategory === category.id
                                    ? 'bg-blue-600 text-white'
                                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                                }`}
                        >
                            {category.name}
                        </button>
                    ))}
                </div>
            </div>

            {loading && (
                <div className="text-center py-4">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
                </div>
            )}

            {error && (
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4">
                    {error}
                </div>
            )}

            <div className="space-y-6">
                {articles.map((article, index) => (
                    <div key={index} className="bg-white rounded-lg shadow-md p-6">
                        <div className="flex justify-between items-start mb-4">
                            <h3 className="text-xl font-semibold text-gray-800">{article.topic}</h3>
                            <div className="flex items-center gap-2">
                                {article.original_article?.url && (
                                    <a
                                        href={article.original_article.url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors duration-200"
                                    >
                                        Read Original Article
                                    </a>
                                )}
                                <button
                                    onClick={() => togglePrompts(index)}
                                    className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded-full transition-colors"
                                >
                                    {showPrompts[index] ? 'Hide Prompts' : 'Show Prompts'}
                                </button>
                                <span className="text-sm text-gray-500">
                                    {new Date(article.timestamp).toLocaleDateString()}
                                </span>
                            </div>
                        </div>
                        
                        {showPrompts[index] ? (
                            <div className="mb-4 p-4 bg-gray-50 rounded-lg">
                                <p className="text-gray-600 whitespace-pre-wrap">{article.text}</p>
                            </div>
                        ) : (
                            <div className="mb-4">
                                <p className="text-gray-600">{cleanText(article.text)}</p>
                            </div>
                        )}

                        {article.image_url && (
                            <div className="relative">
                                <img
                                    src={article.image_url}
                                    alt={article.topic}
                                    className="w-full h-96 object-contain rounded-lg mb-4 shadow-lg hover:scale-105 transition-transform duration-300"
                                />
                            </div>
                        )}
                        
                        {article.original_article && (
                            <div className="mt-4 pt-4 border-t border-gray-200">
                                <div className="flex flex-col space-y-2">
                                    <p className="text-sm text-gray-500">
                                        Source: {article.original_article.source} • {new Date(article.original_article.published_at).toLocaleDateString()}
                                    </p>
                                    <a
                                        href={article.original_article.url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors duration-200"
                                    >
                                        Read Original Article
                                    </a>
                                </div>
                            </div>
                        )}
                        
                        <div className="flex flex-wrap gap-2 mt-4">
                            <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                                Age Group: {article.age_group}
                            </span>
                            <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">
                                Score: {article.combined_score?.toFixed(2)}
                            </span>
                        </div>
                    </div>
                ))}
            </div>

            {!loading && articles.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                    No articles found in this category.
                </div>
            )}
        </div>
    );
};

export default ArticleHistory; 