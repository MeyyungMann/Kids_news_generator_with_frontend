import React, { useState, useEffect } from 'react';
import axios from 'axios';
import FeedbackForm from './FeedbackForm';

const ArticleHistory = () => {
    const [articles, setArticles] = useState([]);
    const [selectedCategory, setSelectedCategory] = useState('all');
    const [selectedAgeGroup, setSelectedAgeGroup] = useState('all');
    const [showFavoritesOnly, setShowFavoritesOnly] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [showPrompts, setShowPrompts] = useState({});  // Track which articles show prompts
    const [deleteLoading, setDeleteLoading] = useState({});  // Track which articles are being deleted
    const [favoriteLoading, setFavoriteLoading] = useState({});  // Track which articles are being favorited
    const [showFeedback, setShowFeedback] = useState({});  // Track which articles show feedback

    const categories = [
        { id: 'all', name: 'All Categories' },
        { id: 'Science', name: 'Science' },
        { id: 'Technology', name: 'Technology' },
        { id: 'Health', name: 'Health' },
        { id: 'Environment', name: 'Environment' },
        { id: 'Economy', name: 'Economy' }
    ];

    const ageGroups = [
        { id: 'all', name: 'All Ages' },
        { id: '3-6', name: 'Ages 3-6' },
        { id: '7-9', name: 'Ages 7-9' },
        { id: '10-12', name: 'Ages 10-12' }
    ];

    useEffect(() => {
        fetchArticles();
    }, [selectedCategory, selectedAgeGroup, showFavoritesOnly]);

    const fetchArticles = async () => {
        try {
            setLoading(true);
            setError(null);
            const response = await axios.get(`http://localhost:8000/api/articles/history?category=${selectedCategory}&age_group=${selectedAgeGroup}`);
            let articlesData = response.data.articles || [];
            
            // Filter favorites if showFavoritesOnly is true
            if (showFavoritesOnly) {
                articlesData = articlesData.filter(article => article.is_favorite === true);
            }
            
            // Sort articles by timestamp (newest first)
            articlesData.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
            
            setArticles(articlesData);
            
            // Initialize showPrompts state as true for all articles
            const newShowPrompts = {};
            articlesData.forEach((_, index) => {
                newShowPrompts[index] = true;  // Set to true by default
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

    const toggleFeedback = (index) => {
        setShowFeedback(prev => ({
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

    const handleToggleFavorite = async (article, index) => {
        try {
            setFavoriteLoading(prev => ({ ...prev, [index]: true }));
            
            // Get the category from the article data
            const category = article.original_article?.category || article.original_article?.original_article?.category;
            
            if (!category) {
                throw new Error('Category not found for article');
            }
            
            // Extract filename from the article data
            const filename = `${article.topic.replace(/[^a-zA-Z0-9]/g, '_')}_${article.timestamp}.json`;
            
            // Call the toggle favorite endpoint with full URL
            const response = await axios.post(`http://localhost:8000/api/articles/${category}/${filename}/favorite`);
            const newFavoriteStatus = response.data.is_favorite;
            
            // Update the article's favorite status in the state
            setArticles(prev => {
                // First update the favorite status
                const updatedArticles = prev.map((a, i) => 
                    i === index ? { ...a, is_favorite: newFavoriteStatus } : a
                );
                
                // If we're showing only favorites and the article was unfavorited, remove it from the list
                if (showFavoritesOnly && !newFavoriteStatus) {
                    return updatedArticles.filter((_, i) => i !== index);
                }
                
                // Sort by timestamp (newest first)
                return updatedArticles.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
            });
            
            // If we're showing all articles and the article was favorited, we might want to show a success message
            if (!showFavoritesOnly && newFavoriteStatus) {
                console.log('Article added to favorites');
            }
            
        } catch (err) {
            console.error('Error toggling favorite:', err);
            alert(err.message || 'Failed to toggle favorite status. Please try again.');
        } finally {
            setFavoriteLoading(prev => ({ ...prev, [index]: false }));
        }
    };

    const handleDelete = async (article, index) => {
        try {
            setDeleteLoading(prev => ({ ...prev, [index]: true }));
            
            // Get the category from the article data
            const category = article.original_article?.category || article.original_article?.original_article?.category;
            
            if (!category) {
                throw new Error('Category not found for article');
            }
            
            // Extract filename from the article data
            const filename = `${article.topic.replace(/[^a-zA-Z0-9]/g, '_')}_${article.timestamp}.json`;
            
            // Call the delete endpoint with full URL
            await axios.delete(`http://localhost:8000/api/articles/${category}/${filename}`);
            
            // Remove the article from the state
            setArticles(prev => prev.filter((_, i) => i !== index));
            
            // Show success message
            alert('Article deleted successfully');
        } catch (err) {
            console.error('Error deleting article:', err);
            alert(err.message || 'Failed to delete article. Please try again.');
        } finally {
            setDeleteLoading(prev => ({ ...prev, [index]: false }));
        }
    };

    const formatDate = (dateString) => {
        if (!dateString) {
            console.log('No date string provided');
            return 'Date not available';
        }

        try {
            // Handle different date formats
            let date;
            if (typeof dateString === 'string') {
                // Try parsing different date formats
                if (dateString.includes('T')) {
                    // ISO format
                    date = new Date(dateString);
                } else if (dateString.includes('_')) {
                    // Custom format YYYYMMDD_HHMMSS
                    const [datePart, timePart] = dateString.split('_');
                    const year = datePart.substring(0, 4);
                    const month = datePart.substring(4, 6);
                    const day = datePart.substring(6, 8);
                    const hour = timePart.substring(0, 2);
                    const minute = timePart.substring(2, 4);
                    const second = timePart.substring(4, 6);
                    date = new Date(`${year}-${month}-${day}T${hour}:${minute}:${second}`);
                } else {
                    // Try standard date parsing
                    date = new Date(dateString);
                }
            } else if (dateString instanceof Date) {
                date = dateString;
            } else {
                console.log('Invalid date format:', dateString);
                return 'Date not available';
            }

            if (isNaN(date.getTime())) {
                console.log('Invalid date value:', dateString);
                return 'Date not available';
            }

            // Format the date with more details
            return date.toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        } catch (error) {
            console.error('Error formatting date:', error, 'Date string:', dateString);
            return 'Date not available';
        }
    };

    const getImageUrl = (imagePath) => {
        if (!imagePath) return null;
        return `http://localhost:8000${imagePath}`;
    };

    return (
        <div className="max-w-4xl mx-auto p-4">
            <div className="mb-6">
                <h2 className="text-2xl font-bold mb-4">Article History</h2>
                <div className="flex flex-wrap items-center gap-4">
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
                    <div className="flex items-center gap-4">
                        <div className="flex items-center gap-2">
                            <label htmlFor="ageGroup" className="text-sm font-medium text-gray-700">Age Group:</label>
                            <select
                                id="ageGroup"
                                value={selectedAgeGroup}
                                onChange={(e) => setSelectedAgeGroup(e.target.value)}
                                className="block w-40 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                            >
                                {ageGroups.map(group => (
                                    <option key={group.id} value={group.id}>
                                        {group.name}
                                    </option>
                                ))}
                            </select>
                        </div>
                        <button
                            onClick={() => setShowFavoritesOnly(!showFavoritesOnly)}
                            className={`px-4 py-2 rounded-full text-sm font-medium transition-colors flex items-center gap-2
                                ${showFavoritesOnly
                                    ? 'bg-yellow-100 text-yellow-700 hover:bg-yellow-200'
                                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                                }`}
                        >
                            <span>{showFavoritesOnly ? '★' : '☆'}</span>
                            {showFavoritesOnly ? 'Show All' : 'Show Favorites'}
                        </button>
                    </div>
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
                                <button
                                    onClick={() => handleToggleFavorite(article, index)}
                                    disabled={favoriteLoading[index]}
                                    className={`px-3 py-1 text-sm rounded-full transition-colors ${
                                        favoriteLoading[index]
                                            ? 'bg-gray-300 cursor-not-allowed'
                                            : article.is_favorite
                                                ? 'bg-yellow-100 text-yellow-700 hover:bg-yellow-200'
                                                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                                    }`}
                                >
                                    {favoriteLoading[index] ? 'Updating...' : article.is_favorite ? '★ Favorited' : '☆ Favorite'}
                                </button>
                                <button
                                    onClick={() => togglePrompts(index)}
                                    className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded-full transition-colors"
                                >
                                    {showPrompts[index] ? 'Hide Original' : 'Show Original'}
                                </button>
                                <button
                                    onClick={() => handleDelete(article, index)}
                                    disabled={deleteLoading[index]}
                                    className={`px-3 py-1 text-sm rounded-full transition-colors ${
                                        deleteLoading[index]
                                            ? 'bg-gray-300 cursor-not-allowed'
                                            : 'bg-red-100 text-red-700 hover:bg-red-200'
                                    }`}
                                >
                                    {deleteLoading[index] ? 'Deleting...' : 'Delete'}
                                </button>
                                <span className="text-sm text-gray-500">
                                    {formatDate(article.timestamp)}
                                </span>
                            </div>
                        </div>

                        {article.image_url && (
                            <div className="mt-4 flex justify-center">
                                <img 
                                    src={getImageUrl(article.image_url)} 
                                    alt={article.topic}
                                    className="w-96 h-96 object-cover rounded-lg shadow-lg"
                                    onError={(e) => {
                                        e.target.onerror = null;
                                        e.target.src = 'path/to/fallback/image.png';
                                    }}
                                />
                            </div>
                        )}
                        
                        {showPrompts[index] ? (
                            <div className="mb-4 p-4 bg-gray-50 rounded-lg">
                                <p className="text-gray-600 whitespace-pre-wrap">{article.text || article.content}</p>
                            </div>
                        ) : (
                            <div className="mb-4">
                                <p className="text-gray-600">{cleanText(article.text || article.content)}</p>
                            </div>
                        )}
                        
                        {article.original_article && (
                            <div className="mt-4 pt-4 border-t border-gray-200">
                                <div className="flex flex-col space-y-2">
                                    <p className="text-sm text-gray-500">
                                        Source: {article.original_article.original_article?.source} • {formatDate(article.original_article.original_article?.published_at)}
                                    </p>
                                    {article.original_article.original_article?.url && (
                                        <a
                                            href={article.original_article.original_article.url}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors duration-200"
                                        >
                                            Read Original Article
                                        </a>
                                    )}
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

                        <div className="mt-6 border-t pt-6">
                            <button
                                onClick={() => toggleFeedback(index)}
                                className="w-full py-3 px-4 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2"
                            >
                                <span className="text-lg">{showFeedback[index] ? '▼' : '▲'}</span>
                                <span className="font-medium">{showFeedback[index] ? 'Hide Feedback Form' : 'Show Feedback Form'}</span>
                            </button>
                            
                            {showFeedback[index] && (
                                <div className="mt-4">
                                    <FeedbackForm
                                        articleId={article.topic.replace(/[^a-zA-Z0-9]/g, '_') + '_' + article.timestamp}
                                        ageGroup={article.age_group}
                                        category={article.original_article?.category || article.original_article?.original_article?.category}
                                    />
                                </div>
                            )}
                        </div>
                    </div>
                ))}
            </div>

            {!loading && articles.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                    {showFavoritesOnly ? 'No favorite articles found.' : 'No articles found in this category.'}
                </div>
            )}
        </div>
    );
};

export default ArticleHistory; 