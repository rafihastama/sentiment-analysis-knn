/* eslint-disable no-unused-vars */
/* eslint-disable react/prop-types */
import React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faHome,
  faBriefcase,
  faPaperPlane,
  faQuestion,
  faImage,
  faCopy,
} from "@fortawesome/free-solid-svg-icons";
import { NavItem, NavLink, Nav } from "reactstrap";
import classNames from "classnames";

import SubMenu from "./SubMenu";

function SideBar({ isOpen, toggle }) {
  return (
    <div className={classNames("sidebar", { "is-open": isOpen })}>
      <div className="sidebar-header">
        <span color="info" onClick={toggle} style={{ color: "#fff" }}>
          &times;
        </span>
        <h3>K-Nearest Neighbor</h3>
        <hr></hr>
      </div>
      <div className="side-menu">
        <Nav vertical className="list-unstyled pb-3">
          <SubMenu title=" Training" icon={faHome} items={submenus[0]} />
          {/* <NavItem>
            <NavLink tag={Link} to={"/about"} style={{ color: "white" }}>
              <FontAwesomeIcon icon={faBriefcase} className="mr-2" />
              About
            </NavLink>
          </NavItem> */}
          <SubMenu title=" Testing" icon={faCopy} items={submenus[1]} />
          {/* <NavItem>
            <NavLink tag={Link} to={"/pages"} style={{ color: "white" }}>
              <FontAwesomeIcon icon={faImage} className="mr-2" />
              Portfolio
            </NavLink>
          </NavItem>
          <NavItem>
            <NavLink tag={Link} to={"/faq"} style={{ color: "white" }}>
              <FontAwesomeIcon icon={faQuestion} className="mr-2" />
              FAQ
            </NavLink>
          </NavItem>
          <NavItem>
            <NavLink tag={Link} to={"/contact"} style={{ color: "white" }}>
              <FontAwesomeIcon icon={faPaperPlane} className="mr-2" />
              Contact
            </NavLink>
          </NavItem> */}
        </Nav>
      </div>
    </div>
  );
}

const submenus = [
  [
    {
      title: "Import",
      target: "",
    },
    {
      title: "Preprocessing",
      target: "preprocessing-train",
    },
    {
      title: "Labeling",
      target: "labeling-train",
    },
    {
      title: "Analysis",
      target: "analysis-train",
    },
    // {
    //   title: "TF-IDF",
    //   target: "tf-idf-train",
    // },
    {
      title: "Modeling",
      target: "modeling-train",
    },
    {
      title: "Evaluate",
      target: "evaluate-train",
    },
  ],
  [
    {
      title: "Split Data",
      target: "split-data-test",
    },
    {
      title: "Preprocessing",
      target: "preprocessing-test",
    },
    {
      title: "Labeling",
      target: "labeling-test",
    },
    {
      title: "Analysis",
      target: "analysis-test",
    },
    {
      title: "Modeling",
      target: "modeling-test",
    },
    {
      title: "Evaluate",
      target: "evaluate-test",
    },
  ],
];

export default SideBar;
