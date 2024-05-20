/* eslint-disable no-unused-vars */
import React, { useEffect, useState } from "react";
import { Card, CardHeader, CardBody, CardTitle, Row, Col } from "reactstrap";
import { PieChart } from "@mui/x-charts/PieChart";
import ReactWordcloud from 'react-wordcloud';
import axios from "axios";

function AnalysisTesting() {
  const [sentimentData, setSentimentData] = useState([]);
  const [wordcloudData, setWordCloudData] = useState([]);

  useEffect(() => {
    fetchSentimentData();
    fetchWordCloudData();
  }, []);

  const fetchSentimentData = async () => {
    try {
      const response = await axios.get("http://localhost:5000/get-sentiment-comparison-testing");
      setSentimentData(response.data);
    } catch (error) {
      console.error("Error fetching sentiment data:", error);
    }
  };

  const fetchWordCloudData = async () => {
    try {
        const response = await axios.get("http://localhost:5000/get-wordcloud-data-testing");
        setWordCloudData(response.data);
    } catch (error) {
        console.error("Error fetching word cloud data:", error);
    }
  }

//   const countSentiments = () => {
//     let positiveCount = 0;
//     let negativeCount = 0;
//     let neutralCount = 0;
//     sentimentData.forEach((item) => {
//       if (item.label === "Positif") {
//         positiveCount += item.value;
//       } else if (item.label === "Negatif") {
//         negativeCount += item.value;
//       } else {
//         neutralCount += item.value;
//       }
//     });
//     return { positiveCount, negativeCount, neutralCount };
//   };

//   const { positiveCount, negativeCount, neutralCount } = countSentiments();

  return (
    <Row>
      <Col sm="6">
        <Card>
          <CardHeader>
            <CardTitle tag="h4">Perbandingan Sentimen</CardTitle>
          </CardHeader>
          <CardBody>
            <PieChart
              series={[
                {
                  data: sentimentData,
                },
              ]}
              width={460}
              height={300}
            />
          </CardBody>
        </Card>
      </Col>

      <Col sm="6">
        <Card>
          <CardHeader>
            <CardTitle tag="h4">Kata yang sering muncul</CardTitle>
          </CardHeader>
          <CardBody>
            <ReactWordcloud
              words={wordcloudData.words}
              options={{
                rotations: 0,
                fontSizes: [20, 100],
                fontFamily: "impact",
                spiral: "archimedean",
                enableTooltip: true,
              }}
            />
          </CardBody>
        </Card>
      </Col>

      <Col sm="6" className="mt-3">
        <Card>
          <CardHeader>
            <CardTitle tag="h4">Wordcloud Positif</CardTitle>
          </CardHeader>
          <CardBody>
            <ReactWordcloud
              words={wordcloudData.positive}
              options={{
                rotations: 0,
                fontSizes: [20, 100],
                fontFamily: "impact",
                spiral: "archimedean",
                enableTooltip: true,
              }}
            />
          </CardBody>
        </Card>
      </Col>

      <Col sm="6" className="mt-3">
        <Card>
          <CardHeader>
            <CardTitle tag="h4">Wordcloud Negatif</CardTitle>
          </CardHeader>
          <CardBody>
            <ReactWordcloud
              words={wordcloudData.negative}
              options={{
                rotations: 0,
                fontSizes: [20, 100],
                fontFamily: "impact",
                spiral: "archimedean",
                enableTooltip: true,
              }}
            />
          </CardBody>
        </Card>
      </Col>
    </Row>
  );
}

export default AnalysisTesting;
