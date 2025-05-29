import React from 'react';

export const About: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-gray-900 mb-8">About Kids News Generator</h1>
      
      <div className="prose max-w-none">
        <p className="text-lg text-gray-700 mb-6">
          Welcome to Kids News Generator, an innovative platform designed to create engaging and educational news content specifically tailored for children. Our mission is to make current events and important topics accessible and understandable for young minds.
        </p>

        <h2 className="text-2xl font-bold text-gray-900 mt-8 mb-4">Features</h2>
        <ul className="list-disc pl-6 space-y-2 text-gray-700">
          <li>Age-appropriate content generation for different age groups (3-6, 7-9, and 10-12)</li>
          <li>Multiple categories including Science, Technology, Health, Environment, and Economy</li>
          <li>Built-in glossary for complex terms</li>
          <li>AI-generated illustrations to enhance understanding</li>
          <li>Multi-language support for diverse audiences</li>
          <li>Comprehensive evaluation metrics for content quality and safety</li>
        </ul>

        <h2 className="text-2xl font-bold text-gray-900 mt-8 mb-4">How It Works</h2>
        <p className="text-gray-700 mb-4">
          Our platform uses advanced AI technology to transform complex topics into child-friendly news articles. The process includes:
        </p>
        <ol className="list-decimal pl-6 space-y-2 text-gray-700">
          <li>Topic analysis and simplification</li>
          <li>Age-appropriate language adaptation</li>
          <li>Educational content enhancement</li>
          <li>Safety and appropriateness verification</li>
          <li>Visual content generation</li>
          <li>Multi-language translation (when requested)</li>
        </ol>

        <h2 className="text-2xl font-bold text-gray-900 mt-8 mb-4">Safety and Quality</h2>
        <p className="text-gray-700 mb-4">
          Every article generated goes through multiple safety checks and quality evaluations:
        </p>
        <ul className="list-disc pl-6 space-y-2 text-gray-700">
          <li>Age appropriateness assessment</li>
          <li>Engagement level evaluation</li>
          <li>Educational value measurement</li>
          <li>Content safety verification</li>
        </ul>

        <h2 className="text-2xl font-bold text-gray-900 mt-8 mb-4">Get Started</h2>
        <p className="text-gray-700">
          Ready to create your first kid-friendly news article? Head to the home page and start generating content by providing a topic and selecting your preferred options. Our system will create a unique, engaging, and educational article tailored to your specifications.
        </p>
      </div>
    </div>
  );
}; 