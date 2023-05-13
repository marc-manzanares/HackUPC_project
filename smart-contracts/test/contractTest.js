const SmartContract = artifacts.require("SmartContract");

contract("SmartContract", () => {
    it('Should update data', async () => {
        const storage = await SmartContract.deployed();
        await storage.updateData("[1.981326871523612, 0.98112312312, 5.1123236789384834]");
        const data = await storage.readData();
        assert(data.toString() === "[1.981326871523612, 0.98112312312, 5.1123236789384834]");
    }),
    it('Should revert duplicated data', async () => {
        const storage = await SmartContract.deployed();
        await storage.updateData("[0.123, 1.123442]");
        try{
            await storage.updateData("[0.123, 1.123442]");
        }
        catch{return;}
        assert.fail("Data is already stored in the Blockchain");
    }); 
});