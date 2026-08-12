var mysql = require('mysql');

// Use a connection pool to automatically handle reconnections
var pool = mysql.createPool({
    connectionLimit: 10, 
    host: process.env.TICKET_OFFICE_MYSQL_HOST,
    port: process.env.TICKET_OFFICE_MYSQL_PORT,
    user: process.env.TICKET_OFFICE_MYSQL_USER,
    password: process.env.TICKET_OFFICE_MYSQL_PASSWORD,
    database: process.env.TICKET_OFFICE_MYSQL_DATABASE
});

var insertEntry = function(name, city, province, region, address, workTime, windowNum) {
    var sql = "INSERT INTO office (name, city, province, region, address, workTime, windowNum) VALUES (?, ?, ?, ?, ?, ?, ?)";
    var values = [name, city, province, region, address, workTime, windowNum];
    
    pool.query(sql, values, function (err, result) {
        if (err) {
            console.error("Failed to insert record:", err);
            return;
        }
        console.log("1 record inserted.");
    });
};

var initData = function(callback) {
    var sql = "CREATE TABLE IF NOT EXISTS office (name VARCHAR(255), city VARCHAR(255), province VARCHAR(255), region VARCHAR(255), address VARCHAR(255), workTime VARCHAR(32), windowNum INT(10))";
    
    pool.query(sql, function (err, result) {
        if (err) {
            console.error("Failed to create table:", err);
            return;
        }
        console.log("Table created");
        insertEntry('Jinqiao Road ticket sales outlets', 'Shanghai', 'Shanghai', 'Pudong New Area', 'Jinqiao Road 1320, Shanghai, Pudong New Area', '08:00-18:00', 1);
        
        if (callback) callback(result);
    });
};

exports.initMysql = function(callback) {
    pool.getConnection(function(err, connection) {
        if (err) {
            console.error("MySQL connection failed:", err);
            return;
        }
        console.log("initMysql连接上数据库啦！");
        connection.release();
        
        initData(function(result){
            if(callback) callback(result);
        });
    });
};

exports.getAll = function(callback) {
    pool.query("SELECT * FROM office", function (err, result) {
        if (err) {
            console.error(err);
            return callback([]);
        }
        callback(result);
    });
};

exports.getSpecificOffices = function(province, city, region, callback) {
    var sql = "SELECT * FROM office WHERE province = ? AND city = ? AND region = ?";
    pool.query(sql, [province, city, region], function (err, result) {
        if (err) {
            console.error(err);
            return callback([]);
        }
        callback(result);
    });
};

exports.addOffice = function(province, city, region, office, callback) {
    insertEntry(office.name, city, province, region, office.address, office.workTime, office.windowNum);
    callback("insert succeed.");
};

exports.deleteOffice = function(province, city, region, officeName, callback) {
    var sql = "DELETE FROM office WHERE name = ? AND province = ? AND city = ? AND region = ?";
    pool.query(sql, [officeName, province, city, region], function (err, result) {
        if (err) {
            console.error(err);
            return callback(err);
        }
        callback(result);
    });
};

exports.updateOffice = function(province, city, region, oldOfficeName, newOffice, callback) {
    var sql = "UPDATE office SET name = ?, address = ?, workTime = ?, windowNum = ? WHERE name = ? AND province = ? AND city = ? AND region = ?";
    var values = [newOffice.name, newOffice.address, newOffice.workTime, newOffice.windowNum, oldOfficeName, province, city, region];
    
    pool.query(sql, values, function (err, result) {
        if (err) {
            console.error(err);
            return callback(err);
        }
        callback(result);
    });
};