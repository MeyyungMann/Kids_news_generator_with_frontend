import React, { useState } from 'react';
import axios from 'axios';

// Tooltip component with different content based on type
const RatingTooltip = ({ type }) => {
  const [showTooltip, setShowTooltip] = useState(false);

  const tooltipContent = {
    age_appropriate: {
      title: "Rating Scale",
      ratings: [
        "1 = Too difficult (not age-appropriate)",
        "2 = Somewhat difficult",
        "3 = Just right",
        "4 = Somewhat easy",
        "5 = Too easy"
      ]
    },
    engagement: {
      title: "Rating Scale",
      ratings: [
        "1 = Too difficult (not age-appropriate)",
        "2 = Somewhat difficult",
        "3 = Just right",
        "4 = Somewhat easy",
        "5 = Too easy"
      ]
    },
    clarity: {
      title: "Rating Scale",
      ratings: [
        "1 = Too difficult (not age-appropriate)",
        "2 = Somewhat difficult",
        "3 = Just right",
        "4 = Somewhat easy",
        "5 = Too easy"
      ]
    }
  };

  return (
    <div className="relative inline-block ml-2">
      <button
        type="button"
        onClick={() => setShowTooltip(!showTooltip)}
        className="text-gray-500 hover:text-gray-700 focus:outline-none"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
          <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
        </svg>
      </button>
      
      {showTooltip && (
        <div className="absolute z-10 w-64 p-4 mt-2 bg-white rounded-lg shadow-lg border border-gray-200">
          <div className="text-sm text-gray-700">
            <h4 className="font-semibold mb-2">{tooltipContent[type].title}:</h4>
            <ul className="space-y-1">
              {tooltipContent[type].ratings.map((rating, index) => (
                <li key={index}>{rating}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

const FeedbackForm = ({ articleId, ageGroup, category }) => {
  const [feedback, setFeedback] = useState({
    age_appropriate: 3,
    engagement: 3,
    clarity: 3,
    age_appropriate_comments: '',
    engagement_comments: '',
    clarity_comments: ''
  });
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState(null);

  const handleRatingChange = (type, value) => {
    setFeedback(prev => ({
      ...prev,
      [type]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    try {
      // Convert age_group to integer if it's a string
      const ageGroupInt = typeof ageGroup === 'string' ? parseInt(ageGroup.split('-')[0]) : ageGroup;

      // Submit all feedback types in a single request with full backend URL
      await axios.post('http://localhost:8000/api/feedback', {
        article_id: articleId,
        age_group: ageGroupInt,
        category: category,
        feedback: [
          {
            feedback_type: 'age_appropriate',
            rating: feedback.age_appropriate,
            comments: feedback.age_appropriate_comments || undefined
          },
          {
            feedback_type: 'engagement',
            rating: feedback.engagement,
            comments: feedback.engagement_comments || undefined
          },
          {
            feedback_type: 'clarity',
            rating: feedback.clarity,
            comments: feedback.clarity_comments || undefined
          }
        ]
      });

      setSubmitted(true);
    } catch (err) {
      setError('Failed to submit feedback. Please try again.');
      console.error('Error submitting feedback:', err);
    }
  };

  if (submitted) {
    return (
      <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded relative" role="alert">
        <strong className="font-bold">Thank you for your feedback!</strong>
        <span className="block sm:inline"> Your input helps us improve our content.</span>
      </div>
    );
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h3 className="text-xl font-semibold mb-4">Help us improve this article</h3>
      <form onSubmit={handleSubmit} className="space-y-4">
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
            <span className="block sm:inline">{error}</span>
          </div>
        )}

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 flex items-center">
            Age Appropriateness
            <RatingTooltip type="age_appropriate" />
          </label>
          <div className="flex space-x-2">
            {[1, 2, 3, 4, 5].map((rating) => (
              <button
                key={rating}
                type="button"
                onClick={() => handleRatingChange('age_appropriate', rating)}
                className={`w-10 h-10 rounded-full ${
                  feedback.age_appropriate === rating
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 text-gray-700'
                }`}
              >
                {rating}
              </button>
            ))}
          </div>
          <textarea
            value={feedback.age_appropriate_comments}
            onChange={(e) => setFeedback(prev => ({ ...prev, age_appropriate_comments: e.target.value }))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            rows="2"
            placeholder="Share your thoughts on age appropriateness..."
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 flex items-center">
            Engagement Level
            <RatingTooltip type="engagement" />
          </label>
          <div className="flex space-x-2">
            {[1, 2, 3, 4, 5].map((rating) => (
              <button
                key={rating}
                type="button"
                onClick={() => handleRatingChange('engagement', rating)}
                className={`w-10 h-10 rounded-full ${
                  feedback.engagement === rating
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 text-gray-700'
                }`}
              >
                {rating}
              </button>
            ))}
          </div>
          <textarea
            value={feedback.engagement_comments}
            onChange={(e) => setFeedback(prev => ({ ...prev, engagement_comments: e.target.value }))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            rows="2"
            placeholder="Share your thoughts on engagement..."
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 flex items-center">
            Clarity
            <RatingTooltip type="clarity" />
          </label>
          <div className="flex space-x-2">
            {[1, 2, 3, 4, 5].map((rating) => (
              <button
                key={rating}
                type="button"
                onClick={() => handleRatingChange('clarity', rating)}
                className={`w-10 h-10 rounded-full ${
                  feedback.clarity === rating
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 text-gray-700'
                }`}
              >
                {rating}
              </button>
            ))}
          </div>
          <textarea
            value={feedback.clarity_comments}
            onChange={(e) => setFeedback(prev => ({ ...prev, clarity_comments: e.target.value }))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            rows="2"
            placeholder="Share your thoughts on clarity..."
          />
        </div>

        <button
          type="submit"
          className="w-full bg-blue-500 text-white py-2 px-4 rounded-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
        >
          Submit Feedback
        </button>
      </form>
    </div>
  );
};

export default FeedbackForm; 