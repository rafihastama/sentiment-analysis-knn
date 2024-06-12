/* eslint-disable no-unused-vars */
/* eslint-disable react/prop-types */
import React from "react"
import { NavItem, NavLink } from "reactstrap"
import { Link, useLocation } from "react-router-dom"

const SubMenu = (props) => {
  const { items } = props
  const location = useLocation()

  return (
    <div>
      {items.map((item, index) => (
        <NavItem key={index} className="pl-4">
          <NavLink 
            tag={Link} 
            to={item.target} 
            style={{ color: "white" }} 
            className={location.pathname === `/${item.target}` ? "active" : ""}
          >
            {item.icon && <span className="mx-2">{item.icon}</span>}
            {item.title}
          </NavLink>
        </NavItem>
      ))}
    </div>
  )
}

export default SubMenu
