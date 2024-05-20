/* eslint-disable no-unused-vars */
import { useState } from 'react'
import './styles/App.css'
import SideBar from './components/Sidebar'
import Content from './content/Content'
import Footer from './components/Footer'

// Bootstrap CSS
import "bootstrap/dist/css/bootstrap.min.css";
// Bootstrap Bundle JS
import "bootstrap/dist/js/bootstrap.bundle.min";

function App() {
  const [sidebarIsOpen, setSidebarOpen] = useState(true);
  const toggleSidebar = () => setSidebarOpen(!sidebarIsOpen)

  return (
    <>
    <div className='App wrapper'>
      <SideBar toggle={toggleSidebar} isOpen={sidebarIsOpen} />
      <Content toggleSidebar={toggleSidebar} sidebarIsOpen={sidebarIsOpen} />
    </div>
    <Footer />
    </>
  )
}

export default App