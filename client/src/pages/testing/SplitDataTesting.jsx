/* eslint-disable no-unused-vars */
import React, { useState, useEffect } from 'react'
import { Button, message } from "antd";
import { SplitCellsOutlined } from "@ant-design/icons";
import TableSplitData from '../../components/TableSplitData';
import axios from "axios";

function SplitDataTesting() {
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    getData()
  }, [])

  const getData = async() => {
    try {
      const response = await axios.get("http://localhost:5000/get-data-testing")
      setData(response.data)
    } catch (error) {
      console.error("Error Fetching Data", error)
    }
  };
  
  const handleProcessClick = async() => {
    setLoading(true)
    try {
      const response = await axios.post("http://localhost:5000/split-data")
      if (response.status === 200) {
        getData()
        message.success("Split Data Success")
      } else {
        message.error("Split Data Error")
      }
    } catch (error) {
      console.error("Error Splitting Data", error)
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteExcel = () => {
    axios
      .delete("http://localhost:5000/delete-all-test")
      .then((response) => {
        if (response.status === 200) {
          setData([]);
          message.success("All tweets deleted successfully");
        } else {
          alert("Failed to delete tweets!");
        }
      })
      .catch((error) => {
        console.error("Error deleting tweets:", error);
        alert("Failed to delete tweets!");
      });
  };

    return (
      <div>
        <h2 style={{ color: "white" }} className="mb-3">
          Split Data
        </h2>
        <Button
          type="primary"
          className="mb-3"
          icon={<SplitCellsOutlined />}
          loading={loading}
          onClick={handleProcessClick}
        >
          Split
        </Button>
        <Button
          type="primary"
          danger
          className="me-3 mx-3"
          onClick={handleDeleteExcel}
        >
          Delete Record
        </Button>
        <TableSplitData data={data} itemsPerPage={5} title={"Testing Data"} />
      </div>
    );
}

export default SplitDataTesting