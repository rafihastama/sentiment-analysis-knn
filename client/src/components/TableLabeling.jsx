/* eslint-disable no-unused-vars */
/* eslint-disable react/prop-types */
import React, { useEffect, useState } from 'react';
import { Table, Pagination, PaginationItem, PaginationLink, Card,
  CardHeader,
  CardBody,
  CardTitle,
  Row,
  Col,
} from 'reactstrap';
import { DownOutlined } from '@ant-design/icons'
import { Dropdown, Typography, Menu } from 'antd';
import axios from 'axios';

const items = [
  { key: '1', label: 'Positif' },
  { key: '2', label: 'Negatif' },
  { key: '3', label: 'Netral' },
];

function TableLabeling({ data, itemsPerPage, title }) {
    const [currentPage, setCurrentPage] = useState(0);
    const [selectedItems, setSelectedItems] = useState({});

    const totalData = data.length;
    const totalPages = Math.ceil(totalData / itemsPerPage);
    const maxDisplayedPages = 5;

    useEffect(() => {
      const fetchSentiments = async () => {
        try {
          const response = await axios.get('http://localhost:5000/get-processed-data');
          const tweets = response.data;
          const updatedSelectedItems = {};
          tweets.forEach(tweet => {
            if (tweet.sentiment !== null) {
              updatedSelectedItems[tweet.id] = tweet.sentiment
              // console.log(updatedSelectedItems[tweet.id])
            }
          });
          setSelectedItems(updatedSelectedItems);
        } catch (error) {
          console.error('Error fetching sentiments:', error)
        }
      };
      fetchSentiments()
    }, [data]);

    const handleClick = (e, index) => {
        e.preventDefault();
        setCurrentPage(index);
    };

    const handleDropdownChange = async (tweetId, label) => {
    try {
        const response = await axios.post('http://localhost:5000/label-sentiment', {
            tweet_id: tweetId,
            sentiment_label: label
        });
        console.log(response.data.message);
        setSelectedItems({ ...selectedItems, [tweetId]: response.data.sentiment });
    } catch (error) {
        console.error('Error labeling sentiment:', error);
    }
    };

    const renderTableData = () => {
      const start = currentPage * itemsPerPage;
      const end = start + itemsPerPage;
      return data.slice(start, end).map((item, index) => {
        if (item.processed_text !== "") {
          return (
            <tr key={index}>
              <td>{item.id}</td>
              <td>{item.username}</td>
              <td>{item.processed_text}</td>
              <td>
                <Dropdown
                  overlay={
                    <Menu>
                      {items.map((menuItem) => (
                        <Menu.Item
                          key={menuItem.key}
                          onClick={() =>
                            handleDropdownChange(item.id, menuItem.label)
                          }
                        >
                          {menuItem.label}
                        </Menu.Item>
                      ))}
                    </Menu>
                  }
                >
                  <Typography.Link>
                    {selectedItems[item.id] !== undefined
                      ? selectedItems[item.id] >= 0.05
                        ? "Positif"
                        : selectedItems[item.id] <= -0.05
                        ? "Negatif"
                        : selectedItems[item.id] > -0.05 &&
                          selectedItems[item.id] < 0.05
                        ? "Netral"
                        : "Value"
                      : "Value"}
                    <DownOutlined />
                  </Typography.Link>
                </Dropdown>
              </td>
            </tr>
          );
        } else {
          return null;
        }
      });
    };

    const renderPagination = () => {
        const paginationItems = [];
        const startPage = Math.max(0, currentPage - Math.floor(maxDisplayedPages / 2));
        const endPage = Math.min(totalPages - 1, startPage + maxDisplayedPages - 1);
        for (let i = startPage; i <= endPage ; i++) {
            paginationItems.push(
                <PaginationItem key={i} active={i === currentPage}>
                    <PaginationLink onClick={(e) => handleClick(e, i)} href="#">
                        {i + 1}
                    </PaginationLink>
                </PaginationItem>
            );
        }
        return paginationItems;
    };

    return (
        <div>
        <Row>
          <Col md="12">
            <Card>
              <CardHeader>
                <CardTitle tag="h4">{title}</CardTitle>
              </CardHeader>
              <CardBody>
                <Table className="tablesorter" responsive>
                  <thead className="text-primary">
                    <tr>
                      <th>No</th>
                      <th>Username</th>
                      <th>Tweet</th>
                      <th>Sentimen</th>
                    </tr>
                  </thead>
                  <tbody>{renderTableData()}</tbody>
                </Table>
              </CardBody>
            </Card>
          </Col>
        </Row>
            <Pagination>
              <PaginationItem disabled={currentPage <= 0}>
                    <PaginationLink
                        onClick={(e) => handleClick(e, 0)}
                        first
                        href="#" />
                </PaginationItem>
                <PaginationItem disabled={currentPage <= 0}>
                    <PaginationLink
                        onClick={(e) => handleClick(e, currentPage - 1)}
                        previous
                        href="#" />
                </PaginationItem>

                {renderPagination()}

                <PaginationItem disabled={currentPage >= totalPages - 1}>
                    <PaginationLink
                        onClick={(e) => handleClick(e, currentPage + 1)}
                        next
                        href="#" />
                </PaginationItem>
                <PaginationItem disabled={currentPage >= totalPages - 1}>
                    <PaginationLink
                        onClick={(e) => handleClick(e, totalPages - 1)}
                        last
                        href="#" />
                </PaginationItem>
            </Pagination>
        </div>
    );
}

export default TableLabeling;
