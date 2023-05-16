import React from "react";
import "./Spinner.css";

export default function Spinner() {
  return (
    <div className="spinner-container">
      <h3>Loading...</h3>
      <div className="loading-spinner">
        <div class="lds-roller">
          <div></div>
          <div></div>
          <div></div>
          <div></div>
          <div></div>
          <div></div>
          <div></div>
          <div></div>
        </div>
      </div>
      <h4>Do not refresh</h4>
    </div>
  );
}
