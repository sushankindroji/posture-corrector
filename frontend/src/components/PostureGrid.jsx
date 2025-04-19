import { useState } from 'react';
import PostureCard from './PostureCard';
import { postures } from '../data/postures';

function PostureGrid() {
  const [activeCategory, setActiveCategory] = useState('All');
  const [searchQuery, setSearchQuery] = useState('');
  
  const categories = ['All', 'General', 'Gym / Athletic', 'Yoga'];
  
  const filteredPostures = postures
    .filter(posture => activeCategory === 'All' || posture.category === activeCategory)
    .filter(posture => 
      searchQuery === '' || 
      posture.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      posture.category.toLowerCase().includes(searchQuery.toLowerCase())
    );

  return (
    <section id="postures" className="py-24 bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="section-title">Choose Your Posture</h2>
          <p className="section-subtitle max-w-2xl mx-auto">
            Select from our comprehensive collection of posture guides
          </p>
        </div>

        <div className="mb-8">
          <div className="max-w-md mx-auto">
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <svg className="h-5 w-5 text-text-muted" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <input
                type="text"
                className="block w-full pl-10 pr-3 py-2 border border-gray-700 rounded-md bg-background-light text-text-primary placeholder-text-muted focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary transition-colors"
                placeholder="Search postures..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
          </div>
        </div>

        <div className="flex flex-wrap justify-center gap-3 mb-12">
          {categories.map(category => (
            <button
              key={category}
              className={`category-btn ${
                activeCategory === category
                  ? 'category-btn-active'
                  : 'category-btn-inactive'
              }`}
              onClick={() => setActiveCategory(category)}
            >
              {category}
            </button>
          ))}
        </div>

        {filteredPostures.length > 0 ? (
          <div className="grid gap-6 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {filteredPostures.map(posture => (
              <PostureCard key={posture.id} posture={posture} />
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <svg className="mx-auto h-12 w-12 text-text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <h3 className="mt-2 text-lg font-medium text-text-primary">No postures found</h3>
            <p className="mt-1 text-text-secondary">Try changing your search or filter.</p>
            <button onClick={() => {setSearchQuery(''); setActiveCategory('All');}} className="mt-4 btn-outline">
              Reset filters
            </button>
          </div>
        )}
      </div>
    </section>
  );
}

export default PostureGrid;