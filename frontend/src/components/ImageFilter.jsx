import React from 'react';

const ImageFilter = ({ 
  clipScore, 
  onFilterChange, 
  sortOrder, 
  onSortChange 
}) => {
  return (
    <div className="bg-white shadow sm:rounded-lg p-4 mb-4">
      <div className="flex flex-col space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700">
            CLIP Similarity Score
          </label>
          <div className="mt-1 flex items-center">
            <input
              type="range"
              min="0"
              max="100"
              value={clipScore}
              onChange={(e) => onFilterChange(Number(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <span className="ml-2 text-sm text-gray-500">{clipScore}%</span>
          </div>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700">
            Sort Images By
          </label>
          <select
            value={sortOrder}
            onChange={(e) => onSortChange(e.target.value)}
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm rounded-md"
          >
            <option value="relevance">Relevance (CLIP Score)</option>
            <option value="newest">Newest First</option>
            <option value="oldest">Oldest First</option>
          </select>
        </div>
      </div>
    </div>
  );
};

export default ImageFilter; 