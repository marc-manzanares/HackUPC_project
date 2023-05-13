const Migrations = artifacts.require("Migrations");

module.exports = function(deployer) {
  deployer.deploy(Migrations, { value: web3.utils.toWei("10", "ether") });
};
