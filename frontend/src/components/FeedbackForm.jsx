import React, { useState } from 'react';
import axios from 'axios';

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
      // Submit all feedback types at once
      await axios.post('/api/feedback', {
        article_id: articleId,
        age_group: ageGroup,
        category: category,
        feedback_type: 'age_appropriate',
        rating: feedback.age_appropriate,
        comments: feedback.age_appropriate_comments
      });

      await axios.post('/api/feedback', {
        article_id: articleId,
        age_group: ageGroup,
        category: category,
        feedback_type: 'engagement',
        rating: feedback.engagement,
        comments: feedback.engagement_comments
      });

      await axios.post('/api/feedback', {
        article_id: articleId,
        age_group: ageGroup,
        category: category,
        feedback_type: 'clarity',
        rating: feedback.clarity,
        comments: feedback.clarity_comments
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
          <label className="block text-sm font-medium text-gray-700">
            Age Appropriateness
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
          <label className="block text-sm font-medium text-gray-700">
            Engagement Level
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
          <label className="block text-sm font-medium text-gray-700">
            Clarity
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