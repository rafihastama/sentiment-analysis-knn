/* eslint-disable no-unused-vars */
/* eslint-disable react/prop-types */
import React from "react"
import {
  ImportOutlined,
  ProfileFilled,
  FileSearchOutlined,
  SplitCellsOutlined,
  FileDoneOutlined,
  FileTextOutlined,
  FileSyncOutlined,
} from "@ant-design/icons"
import { Nav } from "reactstrap"
import classNames from "classnames"

import SubMenu from "./SubMenu"

function SideBar({ isOpen, toggle }) {
  return (
    <div className={classNames("sidebar", { "is-open": isOpen })}>
      <div className="sidebar-header">
        <span color="info" onClick={toggle} style={{ color: "#fff" }}>
          &times
        </span>
        <h3>K-Nearest Neighbour</h3>
        <hr></hr>
      </div>
      <div className="side-menu">
        <Nav vertical className="list-unstyled pb-3">
          <SubMenu items={submenus[0]} />
        </Nav>
      </div>
    </div>
  )
}

const submenus = [
  [
    {
      title: "Import",
      target: "",
      icon: <ImportOutlined />,
    },
    {
      title: "Preprocessing",
      target: "preprocessing",
      icon: <ProfileFilled />,
    },
    {
      title: "Labeling",
      target: "labeling",
      icon: <FileSearchOutlined />,
    },
    {
      title: "Split Data",
      target: "split-data",
      icon: <SplitCellsOutlined />,
    },
    {
      title: "Analysis",
      target: "analysis",
      icon: <FileTextOutlined />,
    },
    {
      title: "Modeling",
      target: "modeling",
      icon: <FileSyncOutlined />,
    },
    {
      title: "Evaluate",
      target: "evaluate",
      icon: <FileDoneOutlined />,
    },
  ],
]

export default SideBar
