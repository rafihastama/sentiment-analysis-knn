/* eslint-disable no-unused-vars */
/* eslint-disable react/prop-types */
import React from "react";
import classNames from "classnames";
import { Container } from "reactstrap";
import { Route, Routes } from "react-router-dom";

import Topbar from "../components/Topbar";
import ImportData from "../pages/training/ImportData";
import LabelingTraining from "../pages/training/LabelingTraining";
import PreprocessingTraining from "../pages/training/PreprocessingTraining";
import AnalysisTraining from "../pages/training/AnalysisTraining";
import TFIDFTraining from "../pages/training/TFIDFTraining";
import ModelingTraining from "../pages/training/ModelingTraining";
import EvaluateTraining from "../pages/training/EvaluateTraining";

import SplitDataTesting from "../pages/testing/SplitDataTesting";
import PreprocessingTesting from "../pages/testing/PreprocessingTesting";
import LabelingTesting from "../pages/testing/LabelingTesting";
import AnalysisTesting from "../pages/testing/AnalysisTesting";
import ModelingTesting from "../pages/testing/ModelingTesting";
import EvaluateTesting from "../pages/testing/EvaluateTesting";

function Content({ sidebarIsOpen, toggleSidebar }) {
    return (
      <Container
        fluid
        className={classNames("content", { "is-open": sidebarIsOpen })}
      >
        <Topbar toggleSidebar={toggleSidebar} />
        <Routes>
          <Route path="/" element={<ImportData />} />
          <Route path="/preprocessing-train" element={<PreprocessingTraining />} />
          <Route path="/labeling-train" element={<LabelingTraining />} />
          <Route path="/analysis-train" element={<AnalysisTraining />} />
          {/* <Route path="/tf-idf-train" element={<TFIDFTraining />} /> */}
          <Route path="/modeling-train" element={<ModelingTraining />} />
          <Route path="/evaluate-train" element={<EvaluateTraining />} />

          <Route path="/split-data-test" element={<SplitDataTesting />} />
          <Route path="/preprocessing-test" element={<PreprocessingTesting />} />
          <Route path="/labeling-test" element={<LabelingTesting />} />
          <Route path="/analysis-test" element={<AnalysisTesting />} />
          <Route path="/modeling-test" element={<ModelingTesting />} />
          <Route path="/evaluate-test" element={<EvaluateTesting />} />
        </Routes>
      </Container>
    );
}

export default Content;
