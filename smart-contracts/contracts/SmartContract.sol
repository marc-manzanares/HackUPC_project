// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.19;

contract SmartContract {
    address payable public owner;

    string[] public data;
    bool private valid;

    constructor() public {
        owner = payable(msg.sender);
    }

    function kill() external {
        require(msg.sender == owner, "Only the owner can kill this contract");
        selfdestruct(owner);
    }

    function updateData(string calldata _data) external {
        valid = verifyData(_data);
        data.push(_data);
    }

    function readData() external view returns(string[] memory) {
        return data;
    }

    function verifyData(string calldata _data) private returns (bool) {
        for (uint i = 0; i < data.length; i++) {
            if (bytes(_data).length != bytes(data[i]).length) {

            }

            if (keccak256(abi.encodePacked(_data)) == keccak256(abi.encodePacked(data[i]))) {
                revert("Data is already stored in the Blockchain");
                return false;
            }
        }
        return true;
        
    }



}