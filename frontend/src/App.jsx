import { useState } from 'react'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import Features from './components/Features'
import PostureGrid from './components/PostureGrid'
import Footer from './components/Footer'
import './App.css'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import PostureDetail from './components/PostureDetail';

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <main>
          <Routes>
            <Route path="/" element={<>
              <Hero />
              <Features />
              <PostureGrid />
            </>} />
            <Route path="/posture/:postureId" element={<PostureDetail />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App