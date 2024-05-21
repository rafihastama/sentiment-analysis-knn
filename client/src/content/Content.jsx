/* eslint-disable no-unused-vars */
/* eslint-disable react/prop-types */
import React from "react";
import classNames from "classnames";
import { Container } from "reactstrap";
import { Route, Routes } from "react-router-dom";

import Topbar from "../components/Topbar";
import ImportData from "../pages/ImportData";
import Preprocessing from "../pages/Preprocessing";
import Labeling from "../pages/Labeling";
import Analysis from "../pages/Analysis";
import SplitData from "../pages/SplitData";
import Modeling from "../pages/Modeling";
import Evaluate from "../pages/Evaluate";

function Content({ sidebarIsOpen, toggleSidebar }) {
    return (
      <Container
        fluid
        className={classNames("content", { "is-open": sidebarIsOpen })}
      >
        <Topbar toggleSidebar={toggleSidebar} />
        <Routes>
          <Route path="/" element={<ImportData />} />
          <Route path="/preprocessing" element={<Preprocessing />} />
          <Route path="/labeling" element={<Labeling />} />
          <Route path="/split-data" element={<SplitData />} />
          <Route path="/analysis" element={<Analysis />} />
          <Route path="/modeling" element={<Modeling />} />
          <Route path="/evaluate" element={<Evaluate />} />
        </Routes>
      </Container>
    );
}

export default Content;
