import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import './App.css';
import Header from './components/Header';
import DataManager from './pages/DataManager';
import HomePage from './pages/HomePage';

const pagerouter = createBrowserRouter([
  {
    path: "/",
    element: <HomePage />,
  },
  {
    path: "/datamanager",
    element: <DataManager />,
  },
]);

function App() {
  return (
    <div className="App">
      <RouterProvider router={pagerouter} />
    </div>  
  );
}

export default App;
