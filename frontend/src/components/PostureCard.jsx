import { Link } from 'react-router-dom';

function PostureCard({ posture }) {
  // Function to get category-specific decorations
  const getCategoryDecoration = () => {
    switch(posture.category) {
      case 'General':
        return {
          iconClass: "text-blue-400",
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5.121 17.804A13.937 13.937 0 0112 16c2.5 0 4.847.655 6.879 1.804M15 10a3 3 0 11-6 0 3 3 0 016 0zm6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          )
        };
      case 'Gym / Athletic':
        return {
          iconClass: "text-amber-400",
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          )
        };
      case 'Yoga':
        return {
          iconClass: "text-emerald-400",
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
          )
        };
      default:
        return {
          iconClass: "text-primary",
          icon: null
        };
    }
  };

  const { iconClass, icon } = getCategoryDecoration();

  return (
    <div className="card group">
      <div className="p-6">
        <div className="flex justify-between items-start mb-4">
          <h3 className="text-lg font-bold text-text-primary group-hover:text-primary transition-colors">{posture.name}</h3>
          <span className={`inline-flex items-center justify-center ${iconClass}`}>
            {icon}
          </span>
        </div>
        <p className="text-sm text-text-muted mb-4">{posture.category}</p>
        <div className="mt-4">
          <Link 
            to={`/posture/${posture.id}`}
            className="details-btn"
          >
            <span>View Details</span>
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </Link>
        </div>
      </div>
    </div>
  );
}

export default PostureCard;