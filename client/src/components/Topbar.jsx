/* eslint-disable no-unused-vars */
/* eslint-disable react/prop-types */
import React, { useState } from "react"
import { MenuFoldOutlined, MenuUnfoldOutlined } from "@ant-design/icons"
import {
  Navbar,
  NavbarBrand,
} from "reactstrap"
import { Button } from "antd"

const Topbar = ({ toggleSidebar }) => {
    const [sidebarIsOpen, setSidebarOpen] = useState(false)

    const handleToggleSidebar = () => {
        setSidebarOpen(!sidebarIsOpen)
        toggleSidebar()
    }

    return (
      <Navbar
        className="navbar shadow-sm p-3 mb-5 bg-white rounded"
        expand="md"
      >
        <Button
          type="text"
          onClick={handleToggleSidebar}
          icon={sidebarIsOpen ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
          style={{
            fontSize: "16px",
            width: 50,
            height: 50,
          }}
        />
        <NavbarBrand href="/">
          <img
            alt="logo"
            src="/react.svg"
            className="react-logo"
            style={{
              height: 40,
              width: 40,
            }}
          />
        </NavbarBrand>
      </Navbar>
    )
}

export default Topbar
