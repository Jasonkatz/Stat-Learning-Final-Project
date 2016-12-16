fs = require( 'fs' );
var events = require( 'events' );
var twitter = require( 'twitter' );
var json2csv = require( 'json2csv' );

var client = new twitter({
    consumer_key: 'DDpJ4IYjnB2c259DewiO01zWK'
    , consumer_secret: 'H7Xnqq2VWCfdCAEUPEb5X8FlqbsqqCxFYmXuce6SMwoQdBMI5X'
    , access_token_key: '42812258-8dvcTHC3vVAvGZKcUgA2ckf8Tc7Wk32nYkhcNbLeX'
    , access_token_secret: '5A0HuZrh7A2yV4atwoX2VoAkruaNrfpVJZa1AvZjwzT28'
});

if ( process.argv.length == 2 ) {
    console.log( 'Enter at least one query item as an argument' );
    return;
}

// Add hashtags to list
var hashtags = [];
for ( var i = 2 ; i < process.argv.length ; ++i ) {
    hashtags.push( process.argv[ i ] );
}

// Create necessary directories in filesystem
if ( !fs.existsSync( './tweets' ) ) {
    fs.mkdirSync( './tweets' );
}
if ( !fs.existsSync( './json' ) ) {
    fs.mkdirSync( './json' );
}
if ( !fs.existsSync( './csv' ) ) {
    fs.mkdirSync( './csv' );
}
for ( var i = 0 ; i < hashtags.length ; ++i ) {
    if ( !fs.existsSync( './tweets/' + hashtags[ i ] ) ) {
        fs.mkdirSync( './tweets/' + hashtags[ i ] );
    }
}

var getTweetsByHashtag = function( hashtag ) {
    var params = {
        q: hashtag
        , count: 100
        , result_type: 'recent'
    };

    var data = {};

    // Create event handler for sending iterative queries for complete data
    var eventEmitter = new events.EventEmitter();
    var queryTwitter = function( max_id ) {

        if ( max_id ) {
            params.max_id = max_id;
        }

        client.get( 'search/tweets', params, function( error, tweets, response ) {
            if ( error ) {
                console.error( 'Error getting data' );
                console.error( response );
                return;
            }

            // We use an object instead of an array here to avoid duplication
            for ( var i = 0 ; i < tweets.statuses.length ; ++i ) {
                data[ tweets.statuses[ i ].id ] = tweets.statuses[ i ];
            }

            console.log( 'Retrieved ' + tweets.statuses.length + ' tweets' );

            // Write tweets to files
            for ( var i = 0 ; i < tweets.statuses.length ; ++i ){
                var obj = tweets.statuses[ i ];
                fs.writeFile( 'tweets/' + hashtag + '/' + obj.id + '.tweet', obj.text, function( err ) {
                    if ( err ) {
                        return console.error( err );
                    }
                });
            }

            // Query the next data if we get new tweets
            if ( !( tweets.statuses.length === 0 ||
                  ( tweets.statuses.length === 1 && data[ tweets.statuses[ 0 ].id ] ) ) ) {
                eventEmitter.emit( 'query', tweets.statuses[ tweets.statuses.length - 1 ].id );
                return;
            }

            console.log( 'Done retrieving tweets' );

            // If we're done getting data, write everything to a csv file
            var pretty = JSON.stringify( data, null, 4 );

            fs.writeFile( 'json/' + hashtag + '.json', pretty, function( err ) {
                if ( err ) {
                    return console.log( err );
                }

                console.log( 'JSON file saved!' );

            });

            createCSV( hashtag, data );
        });

    };

    eventEmitter.on( 'query', queryTwitter );
    eventEmitter.emit( 'query' );
}

for ( var i = 0 ; i < hashtags.length ; ++i ) {
    console.log( 'Retrieving ' + hashtags[ i ] + ' tweets.' );
    getTweetsByHashtag( hashtags[ i ] );
}

var createCSV = function( label, obj ) {
    var tweets = obj;
    var tweetsArray = [];

    // Separate and store hashtags and user_mentions as comma separated string
    for ( var key in tweets ) {
        var tweet = tweets[ key ];
        var hashtagString = '';
        var user_mentionsString = '';

        for ( var j = 0 ; j < tweet.entities.hashtags.length ; ++j ) {
            // Add comma if j not zero
            if ( j ) {
                hashtagString += ', ';
            }
            hashtagString += tweet.entities.hashtags[ j ].text;
        }

        for ( var j = 0 ; j < tweet.entities.user_mentions.length ; ++j ) {
            // Add comma if j not zero
            if ( j ) {
                user_mentionsString += ', ';
            }
            user_mentionsString += tweet.entities.user_mentions[ j ].screen_name;
        }

        tweet.hashtags = hashtagString;
        tweet.user_mentions = user_mentionsString;

        tweetsArray.push( tweet );
    }

    var fields = [
        {
            value: 'id'
        }
        , {
            value: 'user.id'
        }
        , {
            value: 'created_at'
        }
        , {
            value: 'text'
        }
        , {
            value: 'hashtags'
        }
        , {
            value: 'user_mentions'
        }
    ];
    var fieldNames = [
        'Tweet id'
        , 'User id'
        , 'Created at'
        , 'Tweet text'
        , 'Hashtags'
        , 'User mentions'
    ];

    var csv = json2csv( {
        data: tweetsArray
        , fields: fields
        , fieldNames: fieldNames
    });

    fs.writeFile( 'csv/' + label + '.csv', csv, function( err ) {
        if ( err ) {
            throw err;
        }

        console.log( 'CSV file saved!' );
    });
}
