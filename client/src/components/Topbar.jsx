/* eslint-disable no-unused-vars */
/* eslint-disable react/prop-types */
import React, { useState } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faAlignLeft } from "@fortawesome/free-solid-svg-icons";
import {
  Navbar,
  NavbarBrand,
  Button,
  NavbarToggler,
} from "reactstrap";

const Topbar = ({ toggleSidebar }) => {
    const [topbarIsOpen, setTopbarOpen] = useState(true);
    const toggleTopbar = () => setTopbarOpen(!topbarIsOpen);

    return (
        <Navbar
            className="navbar shadow-sm p-3 mb-5 bg-white rounded"
            expand="md"
        >
        <Button onClick={toggleSidebar} style={{ backgroundColor:"#2676f6" }}>
            <FontAwesomeIcon icon={faAlignLeft} />
        </Button>
            <NavbarToggler onClick={toggleTopbar} />
            <NavbarBrand href="/">
            <img
                alt="logo"
                src="/react.svg"
                style={{
                height: 40,
                width: 40
            }} />
            </NavbarBrand>
        </Navbar>
    );
}

export default Topbar;
